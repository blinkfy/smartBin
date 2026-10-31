"""Device-local circle decoding and registration, with an optional helper override.

FTX_ROUNDCODE_DECODER is a JSON argv array for an existing local helper. It reads
one JSON object {imageBase64,mime} on stdin and emits {scanType,rawBase64} on stdout.
Without an override, the bundled Python/ARM64 runtime is used automatically.
Only a WX_CODE result can register the server-issued, short-lived generation task.
"""
import base64
import binascii
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

MAX_IMAGE = 512 * 1024
MAX_OUTPUT = 8192
DECODER_TIMEOUT = 8
_diagnostic = threading.local()
_ERROR_MESSAGES = {
    'DECODER_DISABLED': '本机解码器未启用或解码资源缺失',
    'DECODER_BUSY': '本机解码器正忙，请稍后重试',
    'INVALID_IMAGE': '图片格式或大小不符合本机解码要求',
    'NO_CODE': '本机解码器未读出圆码',
    'WRONG_CODE_TYPE': '本机解码结果不是微信圆码',
    'MULTIPLE_CODES': '图片中检测到多个码',
    'DECODER_FAILED': '本机解码器执行失败',
    'DECODER_UNAVAILABLE': '本机解码器无法运行',
    'DECODER_TIMEOUT': '本机解码超时',
    'DEPENDENCIES_MISSING': '本机解码依赖安装失败，请检查网络及 Python pip',
    'RUNTIME_INTEGRITY_FAILED': '本机解码资源校验失败，请更新完整解码资源',
    'SETUP_BUSY': '本机解码依赖正在初始化，请稍后重试',
    'TASK_MISSING': '后端未返回圆码登记凭证，请确认后端已更新且登记服务可用',
    'TASK_INVALID': '圆码登记凭证无效或已过期，请重新生成',
    'REGISTER_NETWORK': '连接圆码登记接口失败，请检查后端网络',
    'REGISTER_RESPONSE': '圆码登记接口返回异常',
    'INVALID_ROUND_CODE_TASK': '圆码登记凭证无效或已过期，请重新生成',
    'ROUND_CODE_TASK_EXPIRED': '圆码登记任务已过期，请重新生成',
    'ROUND_CODE_TASK_USED': '圆码登记任务已被使用，请重新生成',
    'ROUND_CODE_MAPPING_CONFLICT': '圆码映射存在冲突，请重新生成',
    'ROUND_CODE_STORE_UNAVAILABLE': '后端圆码登记存储暂不可用',
    'ROUND_CODE_TASK_BUSY': '后端圆码登记繁忙，请稍后重试',
}


def _failure(reason, status=None):
    # Only allow-listed diagnostics escape. Never include payloads, tickets,
    # request URLs, subprocess output or arbitrary server/exception messages.
    message = _ERROR_MESSAGES.get(reason, _ERROR_MESSAGES['DECODER_FAILED'])
    if isinstance(status, int) and 100 <= status <= 599:
        message += f'（HTTP {status}）'
    _diagnostic.message = message
    return None


def registration_error():
    """Reason for the current thread's last attempt, without credentials."""
    return getattr(_diagnostic, 'message', '') or '圆码解码登记未完成'



def decoder_command(environ=None):
    settings = os.environ if environ is None else environ
    value = settings.get("FTX_ROUNDCODE_DECODER", "").strip()
    if not value:
        if settings.get("FTX_ROUNDCODE_ENABLED", "1").lower() in ("0", "false", "no"):
            return None
        helper = Path(__file__).resolve().with_name("roundcode_decoder.py")
        assets = helper.parent / "roundcode_assets/runtime.zip"
        return [sys.executable, str(helper), "--stdin-json"] if helper.is_file() and assets.is_file() else None
    try:
        command = json.loads(value)
    except (TypeError, ValueError):
        raise ValueError("FTX_ROUNDCODE_DECODER must be a JSON argv array") from None
    if (not isinstance(command, list) or not 1 <= len(command) <= 32
            or any(not isinstance(arg, str) or not arg or len(arg) > 4096 or "\0" in arg for arg in command)):
        raise ValueError("Invalid local decoder command")
    executable = Path(command[0]).name.lower().removesuffix(".exe")
    if executable in {"java", "javaw", "jar", "sh", "bash", "cmd", "powershell", "pwsh"} or any("unidbg" in arg.lower() for arg in command):
        raise ValueError("Configure a non-Java native decoder helper, without a shell wrapper")
    return command


def server_root(server):
    parsed = urlsplit(server)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Invalid server URL")
    return server.rstrip("/")


def canonical_raw(value):
    if not isinstance(value, str) or not 12 <= len(value) <= 4096 or not re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", value):
        return None
    try:
        data = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error):
        return None
    return value if len(data) >= 8 and base64.b64encode(data).decode("ascii") == value else None


def valid_task(task, now=None):
    if not isinstance(task, dict):
        return False
    timestamp = time.time() * 1000 if now is None else now
    expires = task.get("expiresAt")
    return (isinstance(task.get("id"), str) and re.fullmatch(r"rq_[a-f0-9]{24}", task["id"]) is not None
            and isinstance(task.get("ticket"), str) and re.fullmatch(r"[a-f0-9]{64}", task["ticket"]) is not None
            and isinstance(expires, (int, float)) and not isinstance(expires, bool)
            and math.isfinite(expires) and expires > timestamp)


def task_from_header(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 2048 or not re.fullmatch(r"[A-Za-z0-9_-]+={0,2}", value):
        return None
    try:
        task = json.loads(base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)))
    except (ValueError, UnicodeError, binascii.Error):
        return None
    return task if valid_task(task) else None


def decode_image(image, mime, command=None, timeout=DECODER_TIMEOUT):
    """Bound input, output, execution time and concurrency; never launch Java/shell.

    Configured helpers are local trusted executables. Reader/writer threads prevent
    pipe deadlocks; stdout is killed after 8KiB rather than buffered without limit.
    """
    _diagnostic.message = ""
    command = decoder_command() if command is None else decoder_command({"FTX_ROUNDCODE_DECODER": json.dumps(command)})
    if not command:
        return _failure("DECODER_DISABLED")
    if not isinstance(image, bytes) or not 0 < len(image) <= MAX_IMAGE or mime not in ("image/png", "image/jpeg"):
        return _failure("INVALID_IMAGE")
    builtin = len(command) == 3 and command[1] == str(Path(__file__).resolve().with_name("roundcode_decoder.py"))
    if not _decoder_lock.acquire(blocking=False):
        return _failure("DECODER_BUSY")
    process = None
    try:
        payload = json.dumps({"imageBase64": base64.b64encode(image).decode("ascii"), "mime": mime}).encode("utf-8")
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                   shell=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        output = bytearray()
        overflow = threading.Event()
        def read_output():
            try:
                while True:
                    chunk = process.stdout.read(1024)
                    if not chunk:
                        break
                    if len(output) + len(chunk) > MAX_OUTPUT:
                        overflow.set()
                        process.kill()
                        break
                    output.extend(chunk)
            except (OSError, ValueError):
                overflow.set()
            finally:
                process.stdout.close()
        def write_input():
            try:
                process.stdin.write(payload)
                process.stdin.close()
            except (OSError, ValueError):
                pass
        reader = threading.Thread(target=read_output, daemon=True, name="roundcode-helper-output")
        writer = threading.Thread(target=write_input, daemon=True, name="roundcode-helper-input")
        reader.start(); writer.start()
        if builtin:
            try:
                from .roundcode_decoder import runtime_ready
            except ImportError:
                from roundcode_decoder import runtime_ready
            # Only first-use wheel setup gets a longer budget. Native work has
            # its own 15s watchdog; later calls stay bounded to 20s end-to-end.
            budget = 20 if runtime_ready() else 150
        else:
            budget = max(0.05, min(DECODER_TIMEOUT, timeout))
        process.wait(timeout=budget)
        reader.join(timeout=0.5); writer.join(timeout=0.5)
        if overflow.is_set() or reader.is_alive():
            return _failure("DECODER_FAILED")
        if process.returncode == 124:
            return _failure("DECODER_TIMEOUT")
        result = json.loads(output.decode("utf-8"))
        if isinstance(result, dict) and result.get("error"):
            return _failure(result["error"] if isinstance(result["error"], str) else "DECODER_FAILED")
        if process.returncode != 0:
            return _failure("DECODER_FAILED")
        raw = canonical_raw(result.get("rawBase64")) if isinstance(result, dict) and result.get("scanType") == "WX_CODE" else None
        return raw or _failure("DECODER_FAILED")
    except subprocess.TimeoutExpired:
        return _failure("DECODER_TIMEOUT")
    except (OSError, ValueError, UnicodeError):
        return _failure("DECODER_UNAVAILABLE")
    finally:
        try:
            if process and process.poll() is None:
                process.kill()
                process.wait(timeout=1)
        except (OSError, subprocess.TimeoutExpired):
            pass
        finally:
            _decoder_lock.release()


_decoder_lock = threading.Lock()


def register_task(task, image, mime, server, post, command=None):
    _diagnostic.message = ""
    if not valid_task(task):
        _failure("TASK_INVALID")
        return False
    raw = decode_image(image, mime, command)
    if not raw:
        return False
    if not valid_task(task):
        _failure("TASK_INVALID")
        return False
    try:
        response = post(server_root(server) + "/api/device/wechat-qr/register-generated",
                        json={"taskId": task["id"], "ticket": task["ticket"], "rawBase64": raw}, timeout=8)
        status = response.status_code
        if status == 404:
            _diagnostic.message = '后端缺少圆码登记接口，请更新后端（HTTP 404）'
            return False
        try:
            result = response.json()
        except (ValueError, TypeError):
            _failure("REGISTER_RESPONSE", status)
            return False
        if isinstance(result, dict) and status == 200 and result.get("code") == 0 and isinstance(result.get("data"), dict) and result["data"].get("registered") is True:
            return True
        reason = result.get("reason") if isinstance(result, dict) else None
        _failure(reason if isinstance(reason, str) and reason in _ERROR_MESSAGES else "REGISTER_RESPONSE", status)
        return False
    except Exception:
        _failure("REGISTER_NETWORK")
        return False


def register_generated(response, server, post=None):
    """Call before publishing a refreshed image, outside UI/state locks."""
    _diagnostic.message = ""
    try:
        command = decoder_command()
        if not command:
            _failure("DECODER_DISABLED")
            return False
        header = response.headers.get("X-Round-Code-Task")
        task = task_from_header(header)
        if not task:
            _failure("TASK_INVALID" if header else "TASK_MISSING")
            return False
        mime = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if post is None:
            import requests
            post = requests.post
        return register_task(task, response.content, mime, server, post, command)
    except Exception:
        _failure("DECODER_UNAVAILABLE")
        return False
