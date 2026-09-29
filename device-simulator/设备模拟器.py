#!/usr/bin/env python3
"""
device_simulator.py

功能：
- 支持垃圾桶（bin）与虚拟机器人（robot）两种模式，可在运行时切换
- 垃圾桶模式周期性上报位置、满载率、电量和重量
- 机器人模式轮询后端控制指令，模拟开始、暂停、急停、复位和闭环执行进度
- 生成设备连接二维码，并启动本地状态与控制页：http://localhost:<ui_port>/
"""

import argparse
import atexit
import base64
import io
import json
import os
import queue
import subprocess
import sys
import threading
import time
import random
import secrets
import hashlib
import math
import mimetypes
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote
import requests
import qrcode

# ----------------- 配置与工具 -----------------
DEFAULT_SERVER = "http://localhost:3000"#"https://rgqexbnzzipc.sealosbja.site"
DEFAULT_FRONTEND = "http://localhost:3001"#"https://rgqexbnzzipc.sealosbja.site"  # 前端地址（二维码扫码后应跳转到前端）
DEFAULT_INTERVAL = 15       # 秒
DEFAULT_ROBOT_INTERVAL = 3  # 机器人进度与遥测刷新间隔（秒）
DEFAULT_UI_PORT = 8010
DEFAULT_MODE = "bin"
DEFAULT_COMMAND_INTERVAL = 1.0
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
QR_FILENAME = os.path.join(SCRIPT_DIR, "device_qr.png")
UI_FILENAME = os.path.join(SCRIPT_DIR, "device-simulator-ui.html")
TEXT_TO_IMAGE_FILENAME = os.path.join(SCRIPT_DIR, "textToImage.js")
SORTING_WORKFLOW_PLAYER_FILENAME = os.path.join(SCRIPT_DIR, "sorting-workflow-player.js")
SORTING_ROBOT_ASSETS_DIR = os.path.join(SCRIPT_DIR, "assets", "sorting-robot")
TEXT_TO_IMAGE_PRESETS_DIR = os.path.join(SCRIPT_DIR, "text-to-image-presets")
SDXS_ADAPTER_FILENAME = os.path.join(SCRIPT_DIR, "sdxsAdapter.js")
SDXS_MODEL_DIR = os.path.join(SCRIPT_DIR, "models", "sdxs-512-dreamshaper")
LCM_ADAPTER_FILENAME = os.path.join(SCRIPT_DIR, "lcmTinyAdapter.js")
LCM_MODEL_DIR = os.path.join(SCRIPT_DIR, "models", "lcm-tiny-sd")
TEXT_TO_IMAGE_MODEL_DIRS = {"sdxs-512-dreamshaper": SDXS_MODEL_DIR, "lcm-tiny-sd": LCM_MODEL_DIR}
NODE_MODULES_DIR = os.path.join(SCRIPT_DIR, "node_modules")
ROBOT_IMAGE_FILENAME = os.path.join(SCRIPT_DIR, "images", "lejv.webp")
BIN_IMAGE_FILENAME = os.path.join(SCRIPT_DIR, "images", "trash-bin-1757776021021.jpg")
YOLO_WORKER_FILENAME = os.path.join(SCRIPT_DIR, "scripts", "yolo_worker.py")
DEFAULT_YOLO_MODEL = os.path.abspath(os.path.join(
    SCRIPT_DIR, "..", "runs", "segment", "train7.30", "weights", "best.pt"
))
TOKEN_POLL_INTERVAL = 300  # 每5分钟同步一次 token（秒）

# 两种模拟设备使用独立身份，避免切换模式后复用同一设备名称、token 与状态。
DEFAULT_BIN_DEVICE_ID = 14
DEFAULT_ROBOT_DEVICE_ID = 15
DEFAULT_DEVICE_ID = DEFAULT_BIN_DEVICE_ID
ROBOT_PHASES = (
    (0, "目标识别"),
    (20, "导航至抓取点"),
    (45, "机械臂抓取"),
    (70, "投放至目标桶"),
    (90, "返回待命区"),
)

WASTE_CATEGORIES = {
    "recyclable": {
        "label": "可回收垃圾",
        "capacityKg": 18.0,
        "items": ["塑料瓶", "易拉罐", "纸盒", "报纸", "玻璃瓶"],
    },
    "kitchen": {
        "label": "厨余垃圾",
        "capacityKg": 22.0,
        "items": ["果皮", "剩菜", "菜叶", "蛋壳", "茶叶渣", "土豆", "萝卜", "香蕉", "番茄"],
    },
    "hazardous": {
        "label": "有害垃圾",
        "capacityKg": 12.0,
        "items": ["废电池", "废药板", "过期药品", "胶囊", "灯管", "药品包装", "温度计"],
    },
    "residual": {
        "label": "其他垃圾",
        "capacityKg": 24.0,
        "items": ["餐巾纸", "烟蒂", "陶瓷碎片", "砖块", "石块", "一次性餐具", "尘土"],
    },
}

YOLO_CLASS_MAP = {
    "bottle": ("recyclable", "塑料瓶"),
    "bottle2": ("recyclable", "塑料瓶"),
    "can": ("recyclable", "易拉罐"),
    "papercup": ("recyclable", "纸盒"),
    "paper": ("recyclable", "报纸"),
    "potato": ("kitchen", "土豆"),
    "potatocut": ("kitchen", "土豆"),
    "rabbitcut": ("kitchen", "剩菜"),
    "mooli": ("kitchen", "萝卜"),
    "banana": ("kitchen", "香蕉"),
    "tomato": ("kitchen", "番茄"),
    "battery": ("hazardous", "废电池"),
    "battery1": ("hazardous", "废电池"),
    "battery5": ("hazardous", "废电池"),
    "drug": ("hazardous", "废药板"),
    "drugbag": ("hazardous", "药品包装"),
    "drugbox": ("hazardous", "药品包装"),
    "capsule": ("hazardous", "胶囊"),
    "brick": ("residual", "砖块"),
    "stone": ("residual", "石块"),
    "china": ("residual", "陶瓷碎片"),
}


def map_yolo_detection(class_name):
    mapped = YOLO_CLASS_MAP.get(str(class_name or "").strip().lower())
    if not mapped:
        return None
    return {"category": mapped[0], "item": mapped[1]}

DEPOSIT_PHASES = (
    ("stabilizing", "检测物体稳定", 0.45),
    ("recognizing", "视觉识别与分类", 0.65),
    ("opening", "开启投放舱门", 0.45),
    ("routing", "导流板切换仓位", 0.55),
    ("weighing", "称重传感器采样", 0.45),
    ("stored", "垃圾已落入分类仓", 0.65),
    ("closing", "关闭舱门并复位", 0.45),
)

# Token生成工具
def generate_device_token(device_id):
    """设备端生成token"""
    # 使用设备ID + 时间戳 + 随机数生成唯一token
    timestamp = str(int(time.time()))
    random_part = secrets.token_hex(16)
    raw_token = f"{device_id}_{timestamp}_{random_part}"
    
    # 使用SHA256生成最终token
    token = hashlib.sha256(raw_token.encode()).hexdigest()[:32]
    return token

# 将米转换为经度/纬度的粗略变化（近似）
def meters_to_deg_lat(m):
    return m / 111111.0

def meters_to_deg_lon(m, lat):
    return m / (111111.0 * max(0.0001, abs(math_cos_lat(lat))))

def math_cos_lat(lat):
    # cos(lat in degrees)
    return math.cos(math.radians(lat))


class YoloRecognitionService:
    """通过独立的 yolov8 Conda Python 长驻进程运行真实垃圾分割模型。"""

    def __init__(self, model_path=DEFAULT_YOLO_MODEL, python_path=None):
        self.model_path = os.path.abspath(model_path)
        self.python_path = python_path
        self.process = None
        self.messages = queue.Queue()
        self.stderr_tail = []
        self.lock = threading.Lock()
        self.names = {}

    def _python_candidates(self):
        candidates = [
            self.python_path,
            os.environ.get("YOLO_PYTHON"),
            os.path.join(os.path.expanduser("~"), ".conda", "envs", "yolov8", "python.exe"),
            sys.executable,
        ]
        seen = set()
        for candidate in candidates:
            if not candidate:
                continue
            resolved = os.path.abspath(candidate)
            key = os.path.normcase(resolved)
            if key not in seen and os.path.isfile(resolved):
                seen.add(key)
                yield resolved

    def _read_stdout(self, process):
        for line in process.stdout:
            try:
                self.messages.put(json.loads(line))
            except json.JSONDecodeError:
                continue

    def _read_stderr(self, process):
        for line in process.stderr:
            message = line.strip()
            if message:
                self.stderr_tail.append(message)
                self.stderr_tail = self.stderr_tail[-20:]

    def _next_message(self, timeout):
        try:
            return self.messages.get(timeout=timeout)
        except queue.Empty as error:
            detail = self.stderr_tail[-1] if self.stderr_tail else "YOLO工作进程无响应"
            raise TimeoutError(detail) from error

    def _start(self):
        if self.process and self.process.poll() is None:
            return
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"缺少YOLO模型：{self.model_path}")
        if not os.path.isfile(YOLO_WORKER_FILENAME):
            raise FileNotFoundError(f"缺少YOLO工作脚本：{YOLO_WORKER_FILENAME}")

        last_error = None
        for python_path in self._python_candidates():
            self.messages = queue.Queue()
            self.stderr_tail = []
            creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            process = subprocess.Popen(
                [python_path, "-I", "-u", YOLO_WORKER_FILENAME, "--model", self.model_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=os.path.dirname(os.path.dirname(SCRIPT_DIR)),
                creationflags=creation_flags,
            )
            self.process = process
            threading.Thread(target=self._read_stdout, args=(process,), daemon=True).start()
            threading.Thread(target=self._read_stderr, args=(process,), daemon=True).start()
            try:
                ready = self._next_message(90)
                if ready.get("type") != "ready":
                    raise RuntimeError(ready.get("message") or "YOLO工作进程启动失败")
                self.python_path = python_path
                self.names = ready.get("names") or {}
                return
            except Exception as error:
                last_error = error
                self.close()

        raise RuntimeError(f"无法启动YOLO识别环境：{last_error or '未找到可用Python'}")

    def recognize(self, image_bytes):
        if not image_bytes:
            raise ValueError("识别图片不能为空")
        with self.lock:
            self._start()
            request_id = secrets.token_hex(8)
            payload = {
                "id": request_id,
                "image": base64.b64encode(image_bytes).decode("ascii"),
            }
            try:
                self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
                self.process.stdin.flush()
            except Exception as error:
                self.close()
                raise RuntimeError(f"YOLO工作进程写入失败：{error}") from error

            deadline = time.monotonic() + 90
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("YOLO识别超时")
                message = self._next_message(remaining)
                if message.get("type") == "result" and message.get("id") == request_id:
                    if message.get("error"):
                        raise RuntimeError(message["error"])
                    return message.get("detections") or []

    def close(self):
        process = self.process
        self.process = None
        self.names = {}
        if process and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=3)
            except Exception:
                process.kill()


YOLO_SERVICE = YoloRecognitionService()
atexit.register(YOLO_SERVICE.close)

# ----------------- HTTP 状态服务 -----------------
class StatusHandler(BaseHTTPRequestHandler):
    simulator = None
    connection_bundle = {
        "token": "",
        "scan_url": "",
        "qr_revision": 0,
        "qr_png": b"",
    }
    connection_history = {}
    # shared state between threads
    simulator_state = {
        "device_id": DEFAULT_DEVICE_ID,
        "mode": DEFAULT_MODE,
        "last_report": None,  # dict or None
        "server": None,
        "qr_path": QR_FILENAME,
        "frontend": None,
        "robot": None,
        "last_command": None,
        "paired_user": None,
    }

    def serve_file(self, file_path, content_type):
        if not os.path.exists(file_path):
            self.send_response(404)
            self.end_headers()
            return
        try:
            with open(file_path, "rb") as file_handle:
                data = file_handle.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)
        except Exception as error:
            print(f"读取静态文件失败 {file_path}: {error}")
            self.send_response(500)
            self.end_headers()

    def serve_module_file(self, file_path, cache_control="public, max-age=3600"):
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            self.send_response(404)
            self.end_headers()
            return
        try:
            with open(file_path, "rb") as file_handle:
                data = file_handle.read()
            content_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
            if file_path.endswith((".js", ".mjs")):
                content_type = "text/javascript; charset=utf-8"
            elif file_path.endswith(".wasm"):
                content_type = "application/wasm"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", cache_control)
            self.end_headers()
            self.wfile.write(data)
        except Exception as error:
            print(f"读取前端模块失败 {file_path}: {error}")
            self.send_response(500)
            self.end_headers()

    def serve_large_file(self, file_path, cache_control="public, max-age=31536000, immutable"):
        """流式发送大型ONNX，避免设备模拟器一次性读入数百MB。"""
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            self.send_response(404)
            self.end_headers()
            return
        try:
            file_size = os.path.getsize(file_path)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(file_size))
            self.send_header("Cache-Control", cache_control)
            self.end_headers()
            with open(file_path, "rb") as file_handle:
                while chunk := file_handle.read(1024 * 1024):
                    self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as error:
            print(f"读取SDXS模型失败 {file_path}: {error}")

    def serve_txt2img_vendor(self, path):
        prefix = "/txt2img-vendor/"
        relative_url = unquote(path[len(prefix):])
        if not relative_url or ".." in relative_url.split("/"):
            self.send_response(404)
            self.end_headers()
            return
        file_path = os.path.abspath(os.path.join(NODE_MODULES_DIR, *relative_url.split("/")))
        modules_root = os.path.abspath(NODE_MODULES_DIR)
        if not file_path.startswith(modules_root + os.sep):
            self.send_response(404)
            self.end_headers()
            return

        if (
            relative_url.startswith("web-txt2img/")
            and file_path.endswith(".js")
            and (os.path.exists(file_path) or relative_url == "web-txt2img/dist/adapters/lcm-tiny-sd.js")
        ):
            try:
                source_path = (
                    SDXS_ADAPTER_FILENAME
                    if relative_url == "web-txt2img/dist/adapters/sd-turbo.js"
                    else file_path
                )
                if relative_url == "web-txt2img/dist/adapters/lcm-tiny-sd.js":
                    source_path = LCM_ADAPTER_FILENAME
                with open(source_path, "r", encoding="utf-8") as file_handle:
                    data = file_handle.read()
                data = data.replace(
                    "import('onnxruntime-web/webgpu')",
                    "import('/txt2img-vendor/onnxruntime-web/dist/ort.webgpu.bundle.min.mjs')"
                )
                data = data.replace(
                    "import('onnxruntime-web')",
                    "import('/txt2img-vendor/onnxruntime-web/dist/ort.bundle.min.mjs')"
                )
                data = data.replace(
                    "import('@xenova/transformers')",
                    "import('/txt2img-vendor/@xenova/transformers/dist/transformers.min.js')"
                )
                if relative_url == "web-txt2img/dist/registry.js":
                    data = self.patch_txt2img_registry_for_sdxs(data)
                encoded = data.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/javascript; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(encoded)
                return
            except Exception as error:
                print(f"重写文生图依赖失败 {file_path}: {error}")
                self.send_response(500)
                self.end_headers()
                return

        self.serve_module_file(file_path)

    @staticmethod
    def patch_txt2img_registry_for_sdxs(data):
        replacements = (
            ("import { SDTurboAdapter } from './adapters/sd-turbo.js';", "import { SDXSAdapter } from './adapters/sd-turbo.js';"),
            ("id: 'sd-turbo',", "id: 'sdxs-512-dreamshaper',"),
            ("displayName: 'SD-Turbo (ONNX Runtime Web)',", "displayName: 'SDXS-512-DreamShaper (ONNX Runtime Web)',"),
            ("supportedBackends: ['webgpu', 'wasm'],", "supportedBackends: ['webgpu'],"),
            ("notes: 'Image size 512×512; seed supported.',", "notes: 'SDXS single-step; 512×512; seed supported.',"),
            ("sizeBytesApprox: 2398 * 1024 * 1024,", "sizeBytesApprox: 661 * 1024 * 1024,"),
            ("sizeGBApprox: 2.34,", "sizeGBApprox: 0.65,"),
            ("sizeNotes: 'UNet ~640MB, text_encoder ~1700MB, vae_decoder ~95MB',", "sizeNotes: 'FP16 UNet and TAESD decoder; Text Encoder stays server-side',"),
            ("createAdapter: () => new SDTurboAdapter(),", "createAdapter: () => new SDXSAdapter(),"),
            ("case 'sd-turbo':", "case 'sdxs-512-dreamshaper':"),
            ("return ['webgpu', 'wasm'];", "return ['webgpu'];"),
        )
        for old, new in replacements:
            if old not in data:
                raise RuntimeError("web-txt2img 版本不兼容，无法注册SDXS模型")
            data = data.replace(old, new, 1)
        data = "import { LCMTinyAdapter } from './adapters/lcm-tiny-sd.js';\n" + data
        data = data.replace("const REGISTRY = [", """const REGISTRY = [
    {
        id: 'lcm-tiny-sd',
        displayName: 'LCM-Tiny-SD (ONNX Runtime Web)',
        task: 'text-to-image',
        supportedBackends: ['webgpu'],
        notes: 'LCM few-step; 512×512; seed supported.',
        sizeBytesApprox: 621 * 1024 * 1024,
        sizeGBApprox: 0.61,
        sizeNotes: 'FP16 UNet; shared TAESD; server-side Text Encoder',
        createAdapter: () => new LCMTinyAdapter(),
    },""", 1)
        data = data.replace("case 'sdxs-512-dreamshaper':", "case 'lcm-tiny-sd':\n        case 'sdxs-512-dreamshaper':", 1)
        return data

    def serve_txt2img_model(self, path):
        prefix = "/txt2img-models/"
        model_id, _, relative_url = unquote(path[len(prefix):]).partition("/")
        model_dir = TEXT_TO_IMAGE_MODEL_DIRS.get(model_id)
        if not model_dir:
            self.send_response(404)
            self.end_headers()
            return
        if not relative_url or ".." in relative_url.split("/"):
            self.send_response(404)
            self.end_headers()
            return
        file_path = os.path.abspath(os.path.join(model_dir, *relative_url.split("/")))
        model_root = os.path.abspath(model_dir)
        if not file_path.startswith(model_root + os.sep):
            self.send_response(404)
            self.end_headers()
            return
        self.serve_large_file(file_path)

    def serve_txt2img_preset(self, path):
        prefix = "/txt2img-presets/"
        relative_url = unquote(path[len(prefix):])
        if not relative_url or ".." in relative_url.split("/"):
            self.send_response(404)
            self.end_headers()
            return
        file_path = os.path.abspath(os.path.join(TEXT_TO_IMAGE_PRESETS_DIR, *relative_url.split("/")))
        preset_root = os.path.abspath(TEXT_TO_IMAGE_PRESETS_DIR)
        if not file_path.startswith(preset_root + os.sep):
            self.send_response(404)
            self.end_headers()
            return
        self.serve_module_file(file_path, "public, max-age=31536000, immutable")

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/" or path == "/index.html":
            self.serve_file(UI_FILENAME, "text/html; charset=utf-8")
            return

        if path == "/textToImage.js":
            self.serve_module_file(TEXT_TO_IMAGE_FILENAME, "no-cache")
            return

        if path == "/lcmScheduler.js":
            self.serve_module_file(os.path.join(SCRIPT_DIR, "lcmScheduler.js"), "no-cache")
            return

        if path == "/sorting-workflow-player.js":
            self.serve_module_file(SORTING_WORKFLOW_PLAYER_FILENAME, "no-cache")
            return

        if path == "/sorting-robot/rig.json":
            self.serve_file(os.path.join(SORTING_ROBOT_ASSETS_DIR, "rig.json"), "application/json; charset=utf-8")
            return

        if path == "/sorting-robot/timeline.json":
            self.serve_file(os.path.join(SORTING_ROBOT_ASSETS_DIR, "timeline.json"), "application/json; charset=utf-8")
            return

        if path.startswith("/sorting-robot/layers/"):
            filename = os.path.basename(unquote(path))
            layer_path = os.path.join(SORTING_ROBOT_ASSETS_DIR, "layers", filename)
            if not filename.lower().endswith(".png") or not os.path.isfile(layer_path):
                self.send_response(404)
                self.end_headers()
                return
            self.serve_file(layer_path, "image/png")
            return

        if path.startswith("/txt2img-vendor/"):
            self.serve_txt2img_vendor(path)
            return

        if path.startswith("/txt2img-presets/"):
            self.serve_txt2img_preset(path)
            return

        if path.startswith("/txt2img-models/"):
            self.serve_txt2img_model(path)
            return

        if path == "/qr":
            query = parse_qs(parsed.query)
            requested_revision = str((query.get("revision") or [""])[-1])
            bundle = (
                self.connection_history.get(requested_revision)
                if requested_revision
                else self.connection_bundle
            )
            if not bundle:
                self.send_response(404)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return
            qr_png = bundle.get("qr_png") or b""
            if not qr_png:
                self.send_response(404)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(qr_png)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(qr_png)
            return

        if path == "/robot-image":
            self.serve_file(ROBOT_IMAGE_FILENAME, "image/webp")
            return

        if path == "/bin-image":
            self.serve_file(BIN_IMAGE_FILENAME, "image/jpeg")
            return

        if path == "/status":
            bundle = self.connection_bundle
            state_snapshot = dict(self.simulator_state)
            state_snapshot.update({
                "token": bundle.get("token") or "",
                "scan_url": bundle.get("scan_url") or "",
                "qr_revision": bundle.get("qr_revision") or 0,
            })
            data = json.dumps(state_snapshot, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return

        # Accept backend notifications (status page can be POSTed to /notify)
        if path == "/notify":
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({ 'ok': True }).encode('utf-8'))
            return

        # fallback: 404
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/yolo-recognize":
            length = int(self.headers.get("Content-Length", 0))
            content_type = self.headers.get("Content-Type", "")
            if length <= 0 or length > 8 * 1024 * 1024 or not content_type.startswith("image/"):
                data = json.dumps({
                    "ok": False,
                    "available": True,
                    "message": "请提交不超过8 MB的图片",
                }, ensure_ascii=False).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            try:
                detections = YOLO_SERVICE.recognize(self.rfile.read(length))
                enriched = []
                selected = None
                for detection in detections:
                    mapped = map_yolo_detection(detection.get("className"))
                    item = dict(detection)
                    if mapped:
                        item.update(mapped)
                        if selected is None:
                            selected = item
                    enriched.append(item)
                response = {
                    "ok": selected is not None,
                    "available": True,
                    "model": os.path.basename(YOLO_SERVICE.model_path),
                    "className": selected.get("className") if selected else None,
                    "confidence": selected.get("confidence") if selected else None,
                    "category": selected.get("category") if selected else None,
                    "item": selected.get("item") if selected else None,
                    "detections": enriched,
                    "message": "识别完成" if selected else "未识别到支持的垃圾类别",
                }
                data = json.dumps(response, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
            except Exception as error:
                data = json.dumps({
                    "ok": False,
                    "available": False,
                    "message": str(error),
                }, ensure_ascii=False).encode("utf-8")
                self.send_response(503)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return

        if path == "/txt2img-embedding":
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0 or length > 16 * 1024:
                self.send_response(400)
                self.end_headers()
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if payload.get("model") not in TEXT_TO_IMAGE_MODEL_DIRS:
                    raise ValueError("不支持的文生图模型")
                from sdxs_text_embedding import get_text_to_image_prompt_embedding

                embedding = get_text_to_image_prompt_embedding(payload.get("prompt"), payload["model"])
                data = embedding.astype("float32", copy=False).tobytes(order="C")
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
            except (ValueError, FileNotFoundError) as error:
                data = json.dumps({"ok": False, "message": str(error)}, ensure_ascii=False).encode("utf-8")
                self.send_response(400 if isinstance(error, ValueError) else 503)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception as error:
                print(f"SDXS文本编码失败: {error}")
                data = json.dumps({"ok": False, "message": "SDXS文本编码失败"}, ensure_ascii=False).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            return

        if path in ("/mode", "/control", "/deposit", "/bin/reset"):
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length) if length > 0 else b''
            content_type = self.headers.get('Content-Type', '')
            try:
                if 'application/json' in content_type:
                    payload = json.loads(body.decode('utf-8') if body else '{}')
                else:
                    values = parse_qs(body.decode('utf-8') if body else '')
                    payload = {key: items[-1] for key, items in values.items() if items}
            except Exception:
                payload = {}

            simulator = self.simulator
            if not simulator:
                self.send_response(503)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'ok': False, 'message': '模拟器未初始化'}, ensure_ascii=False).encode('utf-8'))
                return

            try:
                if path == "/mode":
                    result = simulator.set_mode(payload.get('mode'))
                elif path == "/deposit":
                    result = simulator.deposit_waste(payload)
                elif path == "/bin/reset":
                    result = simulator.reset_bin()
                else:
                    result = simulator.apply_robot_command({
                        'id': f"local-{int(time.time() * 1000)}",
                        'action': payload.get('action'),
                        'type': 'local_control',
                        'source': 'simulator_ui'
                    })
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'ok': True, 'data': result}, ensure_ascii=False).encode('utf-8'))
            except Exception as error:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'ok': False, 'message': str(error)}, ensure_ascii=False).encode('utf-8'))
            return
        
        # Accept backend notifications
        if path == "/notify":
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length) if length > 0 else b''
            try:
                payload = json.loads(body.decode('utf-8') if body else '{}')
            except Exception:
                payload = { 'raw': body.decode('utf-8', errors='ignore') }
            # attach to state
            self.simulator_state['last_notify'] = payload
            event_type = payload.get('type') if isinstance(payload, dict) else None
            if event_type == 'user_connected':
                user = payload.get('user') or {}
                self.simulator_state['paired_user'] = {
                    'id': user.get('id'),
                    'username': user.get('username') or '未知用户',
                    'avatar': user.get('avatar'),
                    'connected_at': payload.get('connected_at') or payload.get('timestamp'),
                }
            elif event_type in ('user_disconnected', 'users_disconnected'):
                self.simulator_state['paired_user'] = None
            print(f"[收到后端通知] {payload}")
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({ 'ok': True }).encode('utf-8'))
            return

        # fallback: 404
        self.send_response(404)
        self.end_headers()

# ----------------- 设备模拟器类 -----------------
class DeviceSimulator:
    def __init__(self, server=DEFAULT_SERVER, interval=DEFAULT_INTERVAL,
                 lat=36.06130, lon=120.34600, ui_port=DEFAULT_UI_PORT,
                 frontend=DEFAULT_FRONTEND, fill=45.0, battery=88.0,
                 weight=12.0, telemetry=True, device_id=None,
                 mode=DEFAULT_MODE, command_interval=DEFAULT_COMMAND_INTERVAL,
                 auto_location=False, bin_device_id=DEFAULT_BIN_DEVICE_ID,
                 robot_device_id=DEFAULT_ROBOT_DEVICE_ID,
                 robot_interval=DEFAULT_ROBOT_INTERVAL):
        self.server = server.rstrip("/")
        self.interval = interval
        self.robot_interval = max(1.0, float(robot_interval))
        self.command_interval = max(0.25, float(command_interval))
        self.lat = float(lat)
        self.lon = float(lon)
        self.ui_port = ui_port
        self.fill = float(fill)
        self.battery = float(battery)
        self.weight = float(weight)
        self.telemetry = bool(telemetry)
        self.auto_location = bool(auto_location)
        self._location_attempted = False
        self.mode = mode if mode in ("bin", "robot") else DEFAULT_MODE
        self.device_ids = {
            "bin": int(bin_device_id),
            "robot": int(robot_device_id),
        }
        # 兼容旧的 --device-id：只覆盖启动时所选模式的设备 ID。
        if device_id is not None:
            self.device_ids[self.mode] = int(device_id)
        self.device_id = self.device_ids[self.mode]
        self.robot_status = "idle"
        self.robot_progress = 0.0
        self.robot_phase = "待命"
        self.robot_task_id = None
        self.robot_task_no = None
        self.last_command_id = None
        self._completed_reported = False
        self.deposit_history = []
        self.deposit_state = {
            "status": "idle",
            "phase": "等待投放",
            "phaseKey": "idle",
            "progress": 0,
            "lid": "closed",
            "activeCategory": None,
            "item": None,
            "weightKg": 0,
            "confidence": None,
            "recognition": None,
            "message": "设备就绪",
        }
        initial_weights = {
            "recyclable": max(0.0, self.weight * 0.34),
            "kitchen": max(0.0, self.weight * 0.28),
            "hazardous": max(0.0, self.weight * 0.12),
            "residual": max(0.0, self.weight * 0.26),
        }
        self.bin_compartments = {}
        for key, config in WASTE_CATEGORIES.items():
            capacity = config["capacityKg"]
            current_weight = min(capacity * 0.92, initial_weights[key])
            self.bin_compartments[key] = {
                "key": key,
                "label": config["label"],
                "capacityKg": capacity,
                "weightKg": round(current_weight, 3),
                "fillRate": round(current_weight / capacity * 100, 1),
                "count": 0,
                "full": current_weight / capacity >= 0.85,
            }
        self._state_lock = threading.RLock()
        # 前端基准地址（用于二维码）
        self.frontend = (frontend or DEFAULT_FRONTEND).rstrip("/")
        self._stop = threading.Event()
        self._mode_changed = threading.Event()
        # state for UI
        StatusHandler.simulator = self
        StatusHandler.simulator_state["device_id"] = self.device_id
        StatusHandler.simulator_state["device_ids"] = dict(self.device_ids)
        StatusHandler.simulator_state["mode"] = self.mode
        StatusHandler.simulator_state["server"] = self.server
        StatusHandler.simulator_state["qr_path"] = QR_FILENAME
        StatusHandler.simulator_state["frontend"] = self.frontend
        StatusHandler.simulator_state["token"] = ""
        StatusHandler.simulator_state["scan_url"] = ""
        StatusHandler.simulator_state["qr_revision"] = 0
        StatusHandler.connection_bundle = {
            "token": "",
            "scan_url": "",
            "qr_revision": 0,
            "qr_png": b"",
        }
        StatusHandler.connection_history = {}
        StatusHandler.simulator_state["location_source"] = "fixed"
        StatusHandler.simulator_state["telemetry"] = {
            "fillRate": self.fill,
            "batteryPct": self.battery,
            "weightKg": self.weight,
        }
        self._sync_robot_state()
        self._sync_bin_state()

    def _robot_snapshot(self):
        return {
            "status": self.robot_status,
            "phase": self.robot_phase,
            "progress": round(self.robot_progress, 1),
            "taskId": self.robot_task_id,
            "taskNo": self.robot_task_no,
            "batteryPct": round(self.battery, 1),
            "latitude": round(self.lat, 7),
            "longitude": round(self.lon, 7),
        }

    def _sync_robot_state(self):
        StatusHandler.simulator_state["mode"] = self.mode
        StatusHandler.simulator_state["robot"] = self._robot_snapshot()

    def _bin_snapshot(self):
        compartments = [dict(value) for value in self.bin_compartments.values()]
        total_weight = sum(item["weightKg"] for item in compartments)
        max_fill = max((item["fillRate"] for item in compartments), default=0)
        return {
            "status": dict(self.deposit_state),
            "compartments": compartments,
            "history": list(self.deposit_history[:12]),
            "totalWeightKg": round(total_weight, 2),
            "maxFillRate": round(max_fill, 1),
            "fullAlarm": any(item["full"] for item in compartments),
            "categories": WASTE_CATEGORIES,
        }

    def _sync_bin_state(self):
        snapshot = self._bin_snapshot()
        StatusHandler.simulator_state["bin"] = snapshot
        self.weight = snapshot["totalWeightKg"]
        self.fill = snapshot["maxFillRate"]

    def _set_deposit_phase(self, key, label, progress, **updates):
        with self._state_lock:
            self.deposit_state.update({
                "status": "processing" if key not in ("idle", "error") else key,
                "phaseKey": key,
                "phase": label,
                "progress": progress,
                **updates,
            })
            self._sync_bin_state()

    def deposit_waste(self, payload):
        if self.mode != "bin":
            raise RuntimeError("当前不是垃圾桶设备模式")
        with self._state_lock:
            if self.deposit_state["status"] == "processing":
                raise RuntimeError("投放舱正在处理上一件垃圾")

            category = str((payload or {}).get("category") or "").strip().lower()
            if category not in WASTE_CATEGORIES:
                raise ValueError("请选择正确的垃圾分类")
            config = WASTE_CATEGORIES[category]
            item = str((payload or {}).get("item") or config["items"][0]).strip()
            try:
                weight = round(float((payload or {}).get("weightKg") or 0.2), 3)
            except (TypeError, ValueError):
                raise ValueError("投放重量格式不正确")
            if weight < 0.01 or weight > 5:
                raise ValueError("单次投放重量应在 0.01 到 5 kg 之间")

            compartment = self.bin_compartments[category]
            if compartment["fillRate"] >= 98:
                raise RuntimeError(f"{compartment['label']}仓已满，请先清运")

            image_base64 = (payload or {}).get("image_Base64")
            if not isinstance(image_base64, str) or not image_base64.startswith("data:image/") or ";base64," not in image_base64[:128]:
                raise ValueError("投放图片格式不正确，请重新生成图片")
            if len(image_base64) > 8 * 1024 * 1024:
                raise ValueError("投放图片不能超过8 MB")

            def _bounded_image_number(key, maximum):
                try:
                    return max(0, min(maximum, int((payload or {}).get(key) or 0)))
                except (TypeError, ValueError):
                    return 0

            image_upload_bytes = _bounded_image_number("imageUploadBytes", 8 * 1024 * 1024)
            image_upload_width = _bounded_image_number("imageUploadWidth", 4096)
            image_upload_height = _bounded_image_number("imageUploadHeight", 4096)

            recognition_source = str((payload or {}).get("recognitionSource") or "manual").strip()[:32]
            detected_class = str((payload or {}).get("detectedClass") or "").strip()[:64]
            try:
                provided_confidence = float((payload or {}).get("recognitionConfidence"))
            except (TypeError, ValueError):
                provided_confidence = None
            confidence = (
                round(min(1.0, max(0.0, provided_confidence)), 4)
                if provided_confidence is not None
                else round(random.uniform(0.91, 0.992), 3)
            )
            recognition = {
                "source": recognition_source,
                "detectedClass": detected_class or None,
                "conflict": bool((payload or {}).get("recognitionConflict")),
                "imagePrompt": str((payload or {}).get("imagePrompt") or "").strip()[:500],
                "imagePresetId": str((payload or {}).get("imagePresetId") or "custom").strip()[:64],
                "imageUploadBytes": image_upload_bytes,
                "imageUploadWidth": image_upload_width,
                "imageUploadHeight": image_upload_height,
            }
            deposit_id = f"DEP-{self.device_id}-{int(time.time() * 1000)}-{secrets.token_hex(3)}"
            self.deposit_state.update({
                "status": "processing",
                "phaseKey": "queued",
                "phase": "物体进入识别区",
                "progress": 3,
                "lid": "closed",
                "activeCategory": category,
                "item": item,
                "weightKg": weight,
                "confidence": confidence,
                "recognition": recognition,
                "depositId": deposit_id,
                "message": "等待物体稳定",
            })
            self._sync_bin_state()

        thread = threading.Thread(
            target=self._run_deposit_cycle,
            args=(category, item, weight, confidence, deposit_id, recognition, image_base64),
            daemon=True,
        )
        thread.start()
        return dict(self.deposit_state)

    def _run_deposit_cycle(self, category, item, weight, confidence, deposit_id, recognition, image_base64):
        progress_values = (12, 28, 43, 58, 72, 88, 96)
        try:
            for (phase_key, phase_label, duration), progress in zip(DEPOSIT_PHASES, progress_values):
                updates = {"message": phase_label}
                if phase_key == "opening":
                    updates["lid"] = "opening"
                elif phase_key in ("routing", "weighing", "stored"):
                    updates["lid"] = "open"
                elif phase_key == "closing":
                    updates["lid"] = "closing"
                self._set_deposit_phase(phase_key, phase_label, progress, **updates)
                time.sleep(duration)

                if phase_key == "stored":
                    with self._state_lock:
                        compartment = self.bin_compartments[category]
                        compartment["weightKg"] = round(
                            min(compartment["capacityKg"], compartment["weightKg"] + weight), 3
                        )
                        compartment["fillRate"] = round(
                            compartment["weightKg"] / compartment["capacityKg"] * 100, 1
                        )
                        compartment["count"] += 1
                        compartment["full"] = compartment["fillRate"] >= 85
                        history_entry = {
                            "id": deposit_id,
                            "time": time.strftime("%H:%M:%S"),
                            "item": item,
                            "category": category,
                            "categoryLabel": compartment["label"],
                            "weightKg": weight,
                            "confidence": confidence,
                            "recognition": dict(recognition),
                            "result": "存储正确",
                            "backendStatus": "pending",
                            "backendMessage": "等待上报",
                            "pointsAwarded": 0,
                        }
                        self.deposit_history.insert(0, history_entry)
                        self.deposit_history = self.deposit_history[:30]
                        self._sync_bin_state()
                    threading.Thread(
                        target=self.post_classification,
                        args=(history_entry, image_base64),
                        daemon=True,
                    ).start()

            self._set_deposit_phase(
                "idle", "分类完成", 100, lid="closed", message=f"{item}已存入{WASTE_CATEGORIES[category]['label']}仓"
            )
            time.sleep(1.2)
            self._set_deposit_phase(
                "idle", "等待投放", 0, lid="closed", activeCategory=None,
                item=None, weightKg=0, confidence=None, recognition=None, message="设备就绪"
            )
        except Exception as error:
            self._set_deposit_phase("error", "投放异常", 0, lid="closed", message=str(error))

    def post_classification(self, history_entry, image_base64=None):
        url = f"{self.server}/api/device/{self.device_id}/classification"
        payload = {
            "category": history_entry["categoryLabel"],
            "confidence": history_entry["confidence"],
            "item_name": history_entry["item"],
            "weight_kg": history_entry["weightKg"],
            "image_Base64": image_base64 or None,
            "deposit_id": history_entry["id"],
            "occurred_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "metadata": {
                "simulated": True,
                "device_mode": "bin",
                "category_key": history_entry["category"],
                "classification_result": history_entry["result"],
                "recognition_source": history_entry.get("recognition", {}).get("source"),
                "detected_class": history_entry.get("recognition", {}).get("detectedClass"),
                "recognition_conflict": history_entry.get("recognition", {}).get("conflict", False),
                "image_prompt": history_entry.get("recognition", {}).get("imagePrompt"),
                "image_preset_id": history_entry.get("recognition", {}).get("imagePresetId"),
                "image_upload_bytes": history_entry.get("recognition", {}).get("imageUploadBytes"),
                "image_upload_width": history_entry.get("recognition", {}).get("imageUploadWidth"),
                "image_upload_height": history_entry.get("recognition", {}).get("imageUploadHeight"),
            },
        }
        last_error = None
        for attempt in range(1, 4):
            try:
                response = requests.post(url, json=payload, timeout=10)
                body = response.json() if response.content else {}
                if response.status_code == 200 and body.get("code") == 0:
                    data = body.get("data") or {}
                    points = data.get("pointsInfo") or {}
                    with self._state_lock:
                        history_entry["backendStatus"] = "synced"
                        history_entry["backendMessage"] = body.get("msg") or "已同步"
                        history_entry["hasUser"] = bool(data.get("has_user"))
                        history_entry["pointsAwarded"] = int(points.get("pointsAwarded") or 0)
                        history_entry["duplicate"] = bool(data.get("duplicate"))
                        self._sync_bin_state()
                    print(
                        f"[分类上报] {history_entry['id']} {history_entry['item']} -> "
                        f"{history_entry['categoryLabel']}，积分 +{history_entry['pointsAwarded']}"
                    )
                    return
                last_error = body.get("msg") or f"HTTP {response.status_code}"
            except Exception as error:
                last_error = str(error)
            if attempt < 3:
                time.sleep(attempt)

        with self._state_lock:
            history_entry["backendStatus"] = "failed"
            history_entry["backendMessage"] = last_error or "上报失败"
            self._sync_bin_state()
        print(f"[分类上报失败] {history_entry['id']}: {last_error}")

    def reset_bin(self):
        if self.mode != "bin":
            raise RuntimeError("当前不是垃圾桶设备模式")
        with self._state_lock:
            if self.deposit_state["status"] == "processing":
                raise RuntimeError("投放处理中，暂时不能清空")
            for compartment in self.bin_compartments.values():
                compartment.update({"weightKg": 0, "fillRate": 0, "count": 0, "full": False})
            self.deposit_history = []
            self.deposit_state.update({
                "status": "idle", "phase": "等待投放", "phaseKey": "idle",
                "progress": 0, "lid": "closed", "activeCategory": None,
                "item": None, "weightKg": 0, "confidence": None,
                "recognition": None, "message": "分类仓已清空",
            })
            self._sync_bin_state()
            return self._bin_snapshot()

    def set_mode(self, mode):
        normalized = str(mode or "").strip().lower()
        if normalized not in ("bin", "robot"):
            raise ValueError("mode 必须是 bin 或 robot")
        with self._state_lock:
            self.mode = normalized
            self.device_id = self.device_ids[normalized]
            self.robot_status = "idle"
            self.robot_progress = 0.0
            self.robot_phase = "待命"
            self.robot_task_id = None
            self.robot_task_no = None
            self.last_command_id = None
            self._completed_reported = False
            self._current_token = None
            StatusHandler.simulator_state["device_id"] = self.device_id
            StatusHandler.simulator_state["device_ids"] = dict(self.device_ids)
            StatusHandler.simulator_state["paired_user"] = None
            StatusHandler.simulator_state.pop("token_sync_error", None)
            StatusHandler.connection_bundle = {
                "token": "",
                "scan_url": "",
                "qr_revision": 0,
                "qr_png": b"",
            }
            if self.mode == "robot":
                StatusHandler.simulator_state["telemetry"] = {
                    "fillRate": 0,
                    "batteryPct": round(self.battery, 2),
                    "weightKg": 0,
                    "latitude": self.lat,
                    "longitude": self.lon,
                    "deviceState": "idle",
                    "currentPhase": "待命",
                    "progress": 0,
                    "source": "virtual_robot",
                    "timestamp": int(time.time()),
                }
            else:
                StatusHandler.simulator_state["telemetry"] = {
                    "fillRate": round(self.fill, 2),
                    "batteryPct": round(self.battery, 2),
                    "weightKg": round(self.weight, 2),
                    "latitude": self.lat,
                    "longitude": self.lon,
                    "deviceState": "serving",
                    "source": "bin_simulator",
                    "timestamp": int(time.time()),
                }
            self._sync_robot_state()
        print(f"[模式切换] 当前模式: {self.mode}，设备 ID: {self.device_id}")
        self._mode_changed.set()
        self.send_online()
        return {"mode": self.mode, "device_id": self.device_id, "robot": self._robot_snapshot()}

    def _phase_for_progress(self, progress):
        phase = ROBOT_PHASES[0][1]
        for threshold, label in ROBOT_PHASES:
            if progress >= threshold:
                phase = label
        return phase

    def apply_robot_command(self, command):
        if self.mode != "robot":
            raise RuntimeError("当前不是虚拟机器人模式")

        action = str((command or {}).get("action") or "start").strip().lower()
        with self._state_lock:
            if action in ("start", "resume"):
                if self.robot_progress >= 100 or self.robot_status in ("emergency_stopped", "stopped", "completed"):
                    self.robot_progress = 0.0
                self.robot_status = "running"
                self.robot_phase = self._phase_for_progress(self.robot_progress)
                self._completed_reported = False
            elif action == "pause":
                if self.robot_status == "running":
                    self.robot_status = "paused"
            elif action in ("emergency_stop", "cancel"):
                self.robot_status = "emergency_stopped" if action == "emergency_stop" else "stopped"
                self.robot_phase = "安全停止"
            elif action == "stop":
                self.robot_status = "stopped"
                self.robot_phase = "已停止"
            elif action == "reset":
                self.robot_status = "idle"
                self.robot_progress = 0.0
                self.robot_phase = "待命"
                self.robot_task_id = None
                self.robot_task_no = None
                self._completed_reported = False
            else:
                raise ValueError(f"不支持的机器人指令: {action}")

            self.robot_task_id = (command or {}).get("taskId") or self.robot_task_id
            self.robot_task_no = (command or {}).get("taskNo") or self.robot_task_no
            self.last_command_id = (command or {}).get("id") or self.last_command_id
            StatusHandler.simulator_state["last_command"] = {
                "id": self.last_command_id,
                "action": action,
                "source": (command or {}).get("source"),
                "issuedAt": (command or {}).get("issuedAt"),
            }
            self._sync_robot_state()
            snapshot = self._robot_snapshot()

        print(f"[机器人指令] {action} -> {snapshot['status']} ({snapshot['progress']}%)")
        # 控制动作发生后立即上报一次，避免等待常规遥测周期才同步到 APP。
        self.post_robot_telemetry()
        self.post_execution_result(action=action)
        return snapshot

    def generate_qr(self, token=None):
        # 二维码内容：扫描后前端可以跳转到 server/#/pages/scan/scan?device_id=...
        # token、链接和二维码以同一个版本发布，避免状态页读到更新过程中的混合状态。
        active_token = str(token or getattr(self, '_current_token', None) or '').strip()
        if not active_token:
            raise ValueError("生成二维码前必须先取得有效设备 token")

        with self._state_lock:
            scan_url = (
                f"{self.frontend}/#/pages/scan/scan?device_id={self.device_id}"
                f"&token={active_token}&device_mode={self.mode}"
            )
            image = qrcode.make(scan_url)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            qr_png = buffer.getvalue()
            revision = str(time.time_ns())

            temp_filename = f"{QR_FILENAME}.{revision}.tmp"
            try:
                with open(temp_filename, "wb") as file_handle:
                    file_handle.write(qr_png)
                os.replace(temp_filename, QR_FILENAME)
            finally:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)

            bundle = {
                "token": active_token,
                "scan_url": scan_url,
                "qr_revision": revision,
                "qr_png": qr_png,
            }
            self._current_token = active_token
            StatusHandler.simulator_state.update({
                "token": active_token,
                "scan_url": scan_url,
                "qr_revision": revision,
            })
            StatusHandler.connection_bundle = bundle
            StatusHandler.connection_history[revision] = bundle
            while len(StatusHandler.connection_history) > 2:
                oldest_revision = next(iter(StatusHandler.connection_history))
                StatusHandler.connection_history.pop(oldest_revision, None)

        print("生成二维码，内容：", scan_url)
        return scan_url

    def start_ui_server(self):
        # 启动一个简单 HTTPServer 显示二维码与状态
        addr = ("", self.ui_port)
        try:
            httpd = ThreadingHTTPServer(addr, StatusHandler)
        except OSError as error:
            raise RuntimeError(
                f"状态页端口 {self.ui_port} 已被占用。请先关闭旧的设备模拟器，"
                f"或使用 --ui-port 指定其它端口。原始错误: {error}"
            ) from error

        def run_server():
            print(f"状态页启动：http://localhost:{self.ui_port}/ （可在同一台机器浏览）")
            try:
                httpd.serve_forever()
            except Exception as error:
                print(f"状态页服务异常: {error}")
            finally:
                httpd.server_close()
        t = threading.Thread(target=run_server, daemon=True)
        t.start()
        return t

    def get_real_location(self):
        """启动时执行一次可选的 IP 粗定位，失败后保持启动参数坐标。"""
        if not self.auto_location or self._location_attempted:
            return self.lat, self.lon

        self._location_attempted = True
        try:
            response = requests.get(
                "https://ipinfo.io/json",
                headers={"User-Agent": "Fentouxia-Device-Simulator/1.0"},
                timeout=6,
            )
            response.raise_for_status()
            loc = str((response.json() or {}).get("loc") or "")
            latitude_text, longitude_text = loc.split(",", 1)
            latitude, longitude = float(latitude_text), float(longitude_text)
            if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
                raise ValueError("IP定位返回了无效坐标")
            StatusHandler.simulator_state["location_source"] = "ip"
            print(f"[定位] 启动时IP粗定位成功: {latitude:.6f}, {longitude:.6f}")
            return latitude, longitude
        except Exception as error:
            StatusHandler.simulator_state["location_source"] = "fixed_fallback"
            print(f"[定位] 自动定位不可用，后续不再重试: {error}")

        print(f"[定位] 使用启动坐标: {self.lat:.6f}, {self.lon:.6f}")
        return self.lat, self.lon

    def simulate_movement(self):
        """仅在启动时解析一次可选的自动定位。"""
        try:
            new_lat, new_lon = self.get_real_location()
            # 只在获取到有效坐标时更新
            if new_lat and new_lon and new_lat != 0 and new_lon != 0:
                self.lat = new_lat
                self.lon = new_lon
                print(f"[定位更新] 纬度: {self.lat:.6f}, 经度: {self.lon:.6f}")
            else:
                print("[定位] 未获取到有效坐标，保持当前位置")
        except Exception as e:
            print(f"[定位错误] {e}，保持当前位置")

    def post_location(self):
        url = f"{self.server}/api/device/{self.device_id}/location"
        payload = {
            "latitude": self.lat,
            "longitude": self.lon,
            "timestamp": int(time.time()),
        }
        # optional: add headers, token, etc.
        headers = {"Content-Type": "application/json"}
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=10)
            try:
                server_text = resp.text
            except:
                server_text = str(resp.status_code)
            StatusHandler.simulator_state["last_report"] = {
                "latitude": self.lat,
                "longitude": self.lon,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(payload["timestamp"])),
                "server_response": f"{resp.status_code} {server_text[:200]}"
            }
            print(f"[位置上报] {payload} -> {resp.status_code}")
        except Exception as e:
            StatusHandler.simulator_state["last_report"] = {
                "latitude": self.lat,
                "longitude": self.lon,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time()))),
                "server_response": f"ERROR {e}"
            }
            print("位置上报失败:", e)

    def post_telemetry(self):
        if not self.telemetry:
            return
        url = f"{self.server}/api/device/{self.device_id}/telemetry"
        self.battery = min(100.0, max(0.0, self.battery - random.uniform(0.05, 0.35)))
        with self._state_lock:
            self._sync_bin_state()
            bin_snapshot = self._bin_snapshot()
        payload = {
            "fillRate": round(self.fill, 2),
            "batteryPct": round(self.battery, 2),
            "weightKg": round(self.weight, 2),
            "latitude": self.lat,
            "longitude": self.lon,
            "deviceState": "serving",
            "pointCode": f"P-{str(self.device_id).zfill(3)}",
            "compartments": bin_snapshot["compartments"],
            "fullAlarm": bin_snapshot["fullAlarm"],
            "timestamp": int(time.time()),
        }
        try:
            resp = requests.post(url, json=payload, timeout=10)
            StatusHandler.simulator_state["telemetry"] = {
                **payload,
                "server_response": f"{resp.status_code} {resp.text[:200]}"
            }
            print(f"[telemetry] {payload} -> {resp.status_code}")
        except Exception as e:
            StatusHandler.simulator_state["telemetry"] = {
                **payload,
                "server_response": f"ERROR {e}"
            }
            print("telemetry report failed:", e)

    def advance_robot_state(self):
        completed_now = False
        with self._state_lock:
            if self.robot_status == "running":
                step = random.uniform(9.0, 15.0)
                self.robot_progress = min(100.0, self.robot_progress + step)
                self.robot_phase = self._phase_for_progress(self.robot_progress)
                self.battery = max(0.0, self.battery - random.uniform(0.08, 0.28))
                self.lat += random.uniform(-0.4, 0.4) * meters_to_deg_lat(1)
                self.lon += random.uniform(-0.4, 0.4) * meters_to_deg_lon(1, self.lat)
                if self.robot_progress >= 100:
                    self.robot_progress = 100.0
                    self.robot_status = "completed"
                    self.robot_phase = "闭环任务完成"
                    completed_now = not self._completed_reported
                    self._completed_reported = True
            self._sync_robot_state()
        if completed_now:
            print("[机器人] 闭环任务完成")
            self.post_execution_result(action="complete")

    def post_robot_telemetry(self):
        if not self.telemetry:
            return
        with self._state_lock:
            snapshot = self._robot_snapshot()
        payload = {
            "fillRate": 0,
            "batteryPct": snapshot["batteryPct"],
            "weightKg": 0,
            "latitude": snapshot["latitude"],
            "longitude": snapshot["longitude"],
            "deviceState": snapshot["status"],
            "currentPhase": snapshot["phase"],
            "progress": snapshot["progress"],
            "pointCode": f"ROBOT-{str(self.device_id).zfill(3)}",
            "source": "virtual_robot",
            "timestamp": int(time.time()),
        }
        url = f"{self.server}/api/device/{self.device_id}/telemetry"
        try:
            resp = requests.post(url, json=payload, timeout=10)
            StatusHandler.simulator_state["telemetry"] = {
                **payload,
                "server_response": f"{resp.status_code} {resp.text[:200]}"
            }
            print(f"[机器人遥测] {snapshot['status']} {snapshot['progress']}% -> {resp.status_code}")
            if snapshot.get("taskId") and snapshot.get("status") in ("running", "paused"):
                self.post_execution_result()
        except Exception as error:
            StatusHandler.simulator_state["telemetry"] = {
                **payload,
                "server_response": f"ERROR {error}"
            }
            print("机器人遥测上报失败:", error)

    def post_execution_result(self, action=None):
        with self._state_lock:
            snapshot = self._robot_snapshot()
        status_map = {
            "start": "running",
            "resume": "running",
            "pause": "paused",
            "stop": "cancelled",
            "cancel": "cancelled",
            "emergency_stop": "cancelled",
            "reset": "dispatched",
            "complete": "succeeded",
        }
        status = status_map.get(action, snapshot["status"])
        payload = {
            "taskId": snapshot.get("taskId"),
            "taskNo": snapshot.get("taskNo"),
            "status": status,
            "progress": snapshot["progress"],
            "deviceState": snapshot["status"],
            "phase": snapshot["phase"],
            "message": f"virtual_robot:{action or status}",
            "timestamp": int(time.time()),
        }
        if action == "complete":
            payload["success"] = True
        if action == "emergency_stop":
            payload["faultCode"] = "emergency_stop"
            payload["component"] = "robot_controller"

        try:
            resp = requests.post(
                f"{self.server}/api/device/{self.device_id}/execution-result",
                json=payload,
                timeout=10,
            )
            StatusHandler.simulator_state["last_execution"] = {
                **payload,
                "server_response": f"{resp.status_code} {resp.text[:200]}"
            }
        except Exception as error:
            StatusHandler.simulator_state["last_execution"] = {
                **payload,
                "server_response": f"ERROR {error}"
            }
            print("机器人执行结果上报失败:", error)

    def poll_robot_commands(self):
        def _poll():
            while not self._stop.is_set():
                if self.mode != "robot" or not getattr(self, '_current_token', None):
                    self._stop.wait(self.command_interval)
                    continue

                token = self._current_token
                device_id = self.device_id
                try:
                    response = requests.get(
                        f"{self.server}/api/device/{device_id}/commands",
                        params={"token": token, "limit": 20},
                        timeout=10,
                    )
                    body = response.json() if response.content else {}
                    items = ((body.get("data") or {}).get("items") or []) if body.get("code") == 0 else []
                    if device_id != self.device_id:
                        self._stop.wait(self.command_interval)
                        continue
                    for command in items:
                        command_id = command.get("id")
                        if command_id and command_id == self.last_command_id:
                            result = self._robot_snapshot()
                        else:
                            try:
                                result = self.apply_robot_command(command)
                            except Exception as error:
                                result = {"status": "rejected", "message": str(error)}
                        if command_id:
                            requests.post(
                                f"{self.server}/api/device/{device_id}/commands/{command_id}/ack",
                                params={"token": token},
                                json=result,
                                timeout=10,
                            )
                except Exception as error:
                    StatusHandler.simulator_state["command_error"] = str(error)
                self._stop.wait(self.command_interval)

        thread = threading.Thread(target=_poll, daemon=True)
        thread.start()
        return thread

    def send_online(self):
        # 在启动时生成token并告知后端设备上线
        # 设备端生成token
        device_id = self.device_id
        simulator_mode = self.mode
        token = generate_device_token(device_id)
        expires_at = int(time.time() + 7 * 60)  # 轮询刷新前保留缓冲时间
        
        url = f"{self.server}/api/device/{device_id}/online"
        payload = { 
            'device_id': device_id,
            'timestamp': int(time.time()),
            'callback_url': f'http://localhost:{self.ui_port}',  # 告知后端我们的回调地址
            'token': token,  # 设备端提供token
            'token_expires_at': expires_at * 1000,  # 转换为毫秒
            'source': 'device_simulator',
            'simulator_mode': simulator_mode,
            'device_name': '虚拟机器人' if simulator_mode == 'robot' else '虚拟智能垃圾桶',
            'latitude': self.lat,
            'longitude': self.lon,
        }
        try:
            resp = requests.post(url, json=payload, timeout=10)
            print(f"[上线通知] 设备生成token: {token[:8]}... -> {resp.status_code} {resp.text[:200]}")
            response_body = resp.json() if resp.content else {}
            if resp.status_code == 200 and response_body.get("code") == 0 and device_id == self.device_id:
                backend_token = str(((response_body.get("data") or {}).get("token") or token)).strip()
                self.generate_qr(backend_token)
            else:
                StatusHandler.simulator_state["token_sync_error"] = response_body.get("msg") or f"HTTP {resp.status_code}"
        except Exception as e:
            StatusHandler.simulator_state["token_sync_error"] = str(e)
            print("上线通知失败:", e)

    def poll_token(self):
        # 周期性刷新token并同步到后端（每5分钟）
        def _poll():
            while not self._stop.is_set():
                if self._stop.wait(TOKEN_POLL_INTERVAL):
                    break
                try:
                    device_id = self.device_id
                    # 设备端生成新token
                    new_token = generate_device_token(device_id)
                    expires_at = int(time.time() + 7 * 60)  # 7分钟后过期
                    
                    # 同步到后端
                    url = f"{self.server}/api/device/{device_id}/token"
                    payload = {
                        'token': new_token,
                        'token_expires_at': expires_at * 1000  # 转换为毫秒
                    }
                    
                    r = requests.post(url, json=payload, timeout=10)
                    response_body = r.json() if r.content else {}
                    if r.status_code == 200 and response_body.get('code') == 0 and device_id == self.device_id:
                        backend_token = str(((response_body.get("data") or {}).get("token") or new_token)).strip()
                        self.generate_qr(backend_token)
                        print(f'[token] 设备刷新token: {new_token[:8]}... 并同步到后端')
                    else:
                        print(f'[token] 同步token到后端失败: {r.status_code} {r.text[:160]}')
                        
                except Exception as e:
                    print('[token] 刷新token失败:', e)
                    
        t = threading.Thread(target=_poll, daemon=True)
        t.start()
        return t

    def run(self):
        # start local status server, then generate a token-backed QR code
        self._current_token = None
        self.start_ui_server()
        if self.auto_location:
            self.simulate_movement()
        # send online notification
        try:
            self.send_online()
        except Exception:
            pass
        # start token polling
        self.poll_token()
        self.poll_robot_commands()
        print(f"开始周期性上报，当前模式: {self.mode}，按 Ctrl-C 停止。")
        try:
            while not self._stop.is_set():
                if self.mode == "robot":
                    self.advance_robot_state()
                    self.post_robot_telemetry()
                else:
                    self.post_location()
                    self.post_telemetry()
                wait_interval = self.robot_interval if self.mode == "robot" else self.interval
                self._mode_changed.wait(wait_interval)
                self._mode_changed.clear()
        except KeyboardInterrupt:
            print("收到中断，退出。")
        finally:
            self._stop.set()

# ----------------- CLI -----------------
def main():
    parser = argparse.ArgumentParser(description="设备模拟器：垃圾桶 / 虚拟机器人 + 二维码连接")
    parser.add_argument("--server", "-s", default=DEFAULT_SERVER, help="后端基准 URL (默认 https://rgqexbnzzipc.sealosbja.site/)")
    parser.add_argument("--interval", "-i", type=float, default=DEFAULT_INTERVAL, help="上报间隔(秒)")
    parser.add_argument("--robot-interval", type=float, default=DEFAULT_ROBOT_INTERVAL, help="机器人进度与遥测刷新间隔（默认 3 秒）")
    parser.add_argument("--lat", type=float, default=36.06130, help="初始纬度")
    parser.add_argument("--lon", type=float, default=120.34600, help="初始经度")
    parser.add_argument("--port", type=int, default=DEFAULT_UI_PORT, help="本地状态页端口 (默认 8010)")
    parser.add_argument("--frontend", default=DEFAULT_FRONTEND, help="前端基准 URL，用作二维码跳转地址 (默认 https://rgqexbnzzipc.sealosbja.site/)")
    parser.add_argument("--fill", type=float, default=45.0, help="初始满载率")
    parser.add_argument("--battery", type=float, default=88.0, help="初始电量百分比")
    parser.add_argument("--weight", type=float, default=12.0, help="初始重量 kg")
    parser.add_argument("--no-telemetry", action="store_true", help="停止上报设备遥测")
    parser.add_argument("--device-id", type=int, default=None, help="兼容参数：覆盖启动模式使用的设备 ID")
    parser.add_argument("--bin-device-id", type=int, default=DEFAULT_BIN_DEVICE_ID, help="虚拟垃圾桶设备 ID（默认 14）")
    parser.add_argument("--robot-device-id", type=int, default=DEFAULT_ROBOT_DEVICE_ID, help="虚拟机器人设备 ID（默认 15）")
    parser.add_argument("--mode", choices=("bin", "robot"), default=DEFAULT_MODE, help="模拟模式：bin 或 robot")
    parser.add_argument("--command-interval", type=float, default=DEFAULT_COMMAND_INTERVAL, help="机器人指令轮询间隔（秒）")
    parser.add_argument("--auto-location", action="store_true", default=False, help="启动时执行一次IP粗定位；默认直接使用 --lat/--lon")
    args = parser.parse_args()

    sim = DeviceSimulator(server=args.server, interval=args.interval,
                          lat=args.lat, lon=args.lon, ui_port=args.port,
                           frontend=args.frontend, fill=args.fill,
                           battery=args.battery, weight=args.weight,
                           telemetry=not args.no_telemetry,
                           device_id=args.device_id, mode=args.mode,
                           command_interval=args.command_interval,
                           auto_location=args.auto_location,
                           bin_device_id=args.bin_device_id,
                           robot_device_id=args.robot_device_id,
                           robot_interval=args.robot_interval)
    sim.run()

if __name__ == "__main__":
    main()
