"""Non-Java ARM64 QBar adapter for an isolated, time-limited helper process.

The caller supplies verified runtime resources and Python dependencies. This
module never resolves scenes, contacts services, or logs decoded content.
"""
from __future__ import annotations

import base64
import contextlib
from io import BytesIO
import importlib
import inspect
import logging
import os
from pathlib import Path
import pkgutil
import shutil
import struct
import sys
import warnings


def _close_emulator(emu):
    for name in ("close", "destroy"):
        close = getattr(emu, name, None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
            return
    # The pinned upstream Emulator has no close method. Close its tracked OS
    # file descriptors; Unicorn releases its memory at subprocess termination.
    pcb = emu.get_pcb()
    files = list(getattr(pcb, "_fds", {}).values())
    mappings = getattr(emu.memory, "_MemoryMap__file_map_addr", {})
    files.extend(mapping[2] for mapping in mappings.values())
    protected = {stream.fileno() for stream in (sys.stdin, sys.stdout, sys.stderr)}
    descriptors = {getattr(item, "descriptor", None) for item in files}
    for descriptor in descriptors:
        if isinstance(descriptor, int) and descriptor > 2 and descriptor not in protected:
            try:
                os.close(descriptor)
            except OSError:
                pass


def decode_bytes(image_bytes: bytes, runtime_root: Path, work_root: Path) -> dict:
    """Decode one bounded PNG/JPEG using private per-call VFS resources.

    Returns only scanType/rawBase64 for exactly one WX_CODE, or an error code.
    The caller must use a distinct work_root and a disposable subprocess for
    every call, enforce a timeout, and delete work_root after the process exits.
    """
    if not isinstance(image_bytes, bytes) or not image_bytes or len(image_bytes) > 512 * 1024:
        return {"error": "INVALID_IMAGE"}
    try:
        from PIL import Image
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(image_bytes)) as image:
                width, height = image.size
                if (image.format not in {"PNG", "JPEG"} or width < 1 or height < 1
                        or max(width, height) > 1600 or width * height > 4_000_000):
                    return {"error": "INVALID_IMAGE"}
                grayscale = image.convert("L")
                low, high = grayscale.getextrema()
                if low == high:
                    return {"error": "NO_CODE"}
                gray = grayscale.tobytes()
    except (OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        return {"error": "INVALID_IMAGE"}

    runtime_root = Path(runtime_root).resolve()
    work_root = Path(work_root).resolve()
    if (work_root == runtime_root or runtime_root in work_root.parents
            or work_root in runtime_root.parents):
        return {"error": "INVALID_WORK_ROOT"}
    vfs_root = work_root / "vfs"
    # Copy system resources rather than hard-linking: even accidental guest
    # writes must never mutate the shared, verified runtime resources.
    try:
        work_root.mkdir(parents=True, exist_ok=True)
        vfs_root.mkdir(exist_ok=False)
        shutil.copytree(runtime_root / "vfs/system", vfs_root / "system")
        for directory in ("proc", "dev", "data", "tmp", "sdcard"):
            (vfs_root / directory).mkdir(exist_ok=True)
    except OSError:
        return {"error": "RUNTIME_UNAVAILABLE"}

    previous_logging = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        with open(os.devnull, "w", encoding="utf-8") as quiet:
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                return _decode_gray(gray, width, height, runtime_root, vfs_root)
    except (Exception, SystemExit):
        # Guest or loader exceptions can contain raw byte arrays. Keep the
        # public error bounded and free of payloads and host filesystem paths.
        return {"error": "DECODER_FAILED"}
    finally:
        logging.disable(previous_logging)


def _decode_gray(gray: bytes, width: int, height: int, runtime_root: Path, vfs_root: Path) -> dict:
    from androidemu.const import emu_const
    from androidemu.emulator import Emulator
    from androidemu.internal.elf_reader import ELFReader
    from androidemu.internal.modules import Modules
    from androidemu.java.classes.array import Array, ByteArray
    from androidemu.java.classes.string import String
    from androidemu.java.helpers.native_method import native_method
    from androidemu.java.java_class_def import JavaClassDef
    from androidemu.java.java_field_def import JavaFieldDef
    from androidemu.java.java_method_def import java_method_def
    from androidemu.java.jni_env import JNIEnv
    from androidemu.java.jni_ref import jclass
    from unicorn import UC_PROT_READ, UC_PROT_WRITE

    # The upstream preloader derives a Python package name from Windows paths.
    def preload_classes(emu):
        package = importlib.import_module("androidemu.java.classes")
        classes = set()
        for _, name, _ in pkgutil.iter_modules(package.__path__):
            module = importlib.import_module("androidemu.java.classes." + name)
            classes.update(cls for _, cls in inspect.getmembers(module, inspect.isclass)
                           if type(cls) is JavaClassDef)
        for cls in classes:
            emu.java_classloader.add_class(cls)
        from androidemu.java.java_classloader import JavaClassLoader
        emu.java_classloader.add_class(JavaClassLoader)

    Emulator._Emulator__add_classes = preload_classes

    # ELF symbol-table entry zero is deliberately undefined, not a missing API.
    original_symval = Modules._elf_get_symval
    def symbol_value(self, load_bias, symbol):
        return 0 if not symbol["name"] else original_symval(self, load_bias, symbol)
    Modules._elf_get_symval = symbol_value

    # Android's legacy soinfo name field is 128 bytes. Host cache paths can be
    # longer; shorten only the serialized name and retain the path for file I/O.
    original_soinfo64 = ELFReader._ELFReader__write_soinfo64
    def write_soinfo64(self, mu, load_base, load_bias, info_base):
        filename = self._ELFReader__filename
        basename = Path(filename).name
        if len(basename.encode("utf-8")) >= 128:
            raise ValueError("Runtime library basename exceeds soinfo capacity")
        try:
            self._ELFReader__filename = basename
            return original_soinfo64(self, mu, load_base, load_bias, info_base)
        finally:
            self._ELFReader__filename = filename
    ELFReader._ELFReader__write_soinfo64 = write_soinfo64

    def pyarray(env, index):
        return JNIEnv.jobject_to_pyobject(env.get_reference(index)).get_py_items()

    @native_method
    def get_byte_elements(self, mu, env, index, is_copy):
        if is_copy:
            mu.mem_write(is_copy, b"\x01")
        data = bytes(pyarray(self, index))
        address = self._emu.memory.map(0, len(data) + 4, UC_PROT_READ | UC_PROT_WRITE)
        mu.mem_write(address, len(data).to_bytes(4, "little") + data)
        return address + 4

    @native_method
    def release_byte_elements(self, mu, env, index, address, mode):
        if not address:
            return
        length = int.from_bytes(mu.mem_read(address - 4, 4), "little")
        if mode != 2:
            pyarray(self, index)[:] = mu.mem_read(address, length)
        if mode != 1:
            self._emu.memory.unmap(address - 4, length + 4)

    @native_method
    def set_byte_region(self, mu, env, index, start, length, source):
        pyarray(self, index)[start:start + length] = mu.mem_read(source, length)

    @native_method
    def get_int_elements(self, mu, env, index, is_copy):
        if is_copy:
            mu.mem_write(is_copy, b"\x01")
        items = pyarray(self, index)
        data = struct.pack("<" + "i" * len(items), *items)
        address = self._emu.memory.map(0, len(data) + 4, UC_PROT_READ | UC_PROT_WRITE)
        mu.mem_write(address, len(data).to_bytes(4, "little") + data)
        return address + 4

    @native_method
    def release_int_elements(self, mu, env, index, address, mode):
        if not address:
            return
        length = int.from_bytes(mu.mem_read(address - 4, 4), "little")
        if mode != 2:
            pyarray(self, index)[:] = struct.unpack("<" + "i" * (length // 4),
                                                  mu.mem_read(address, length))
        if mode != 1:
            self._emu.memory.unmap(address - 4, length + 4)

    @native_method
    def get_int_region(self, mu, env, index, start, length, target):
        mu.mem_write(target, struct.pack("<" + "i" * length,
                                        *pyarray(self, index)[start:start + length]))

    @native_method
    def new_int_array(self, mu, env, length):
        return Array([0] * length)

    @native_method
    def get_java_vm(self, mu, env, target):
        mu.mem_write(target, self._emu.java_vm.address_ptr.to_bytes(8, "little"))
        return 0

    for name, func in {
        "get_byte_array_elements": get_byte_elements,
        "release_byte_array_elements": release_byte_elements,
        "set_byte_array_region": set_byte_region,
        "get_int_array_elements": get_int_elements,
        "release_int_array_elements": release_int_elements,
        "get_int_array_region": get_int_region,
        "new_int_array": new_int_array,
        "get_java_vm": get_java_vm,
    }.items():
        setattr(JNIEnv, name, func)

    model_fields = {
        "detectModelVersion": "Ljava/lang/String;",
        "superResolutionModelVersion": "Ljava/lang/String;",
        "detect_model_bin_path_": "Ljava/lang/String;",
        "detect_model_param_path_": "Ljava/lang/String;",
        "superresolution_model_bin_path_": "Ljava/lang/String;",
        "superresolution_model_param_path_": "Ljava/lang/String;",
        "qbar_segmentation_model_path_": "Ljava/lang/String;",
        "enable_seg": "Z",
    }

    def make_class(name, field_types):
        def initialize(self):
            for key, signature in field_types.items():
                setattr(self, key, String("") if signature == "Ljava/lang/String;" else
                        ByteArray(bytearray(1024)) if signature == "[B" else 0)
        return JavaClassDef(name.rsplit("$", 1)[-1], (), {"__init__": initialize},
                            jvm_name="com/tencent/qbar/" + name,
                            jvm_fields=[JavaFieldDef(k, v, False)
                                        for k, v in field_types.items()])

    Model = make_class("QbarNative$QbarAiModelParam", model_fields)
    Result = make_class("QbarNative$QBarResultJNI", {
        "charset": "Ljava/lang/String;", "typeName": "Ljava/lang/String;",
        "data": "[B", "typeID": "I",
    })
    extra_classes = [make_class(name, fields) for name, fields in {
        "WxQbarNative": {},
        "QbarNative$QBarCodeDetectInfo": {"prob": "F", "readerId": "I"},
        "QbarNative$QBarZoomInfo": {"isZoom": "Z", "zoomFactor": "F"},
        "QbarNative$QBarPoint": {"point_cnt": "I", **{axis + str(i): "F"
                                  for axis in "xy" for i in range(4)}},
        "QbarNative$QBarReportMsg": {"qrcodeVersion": "I"},
        "WxQbarNative$QBarReportMsg": {
            **{k: "Ljava/lang/String;" for k in ["binaryMethod", "charsetMode", "ecLevel", "scaleList"]},
            **{k: "I" for k in ["decodeTime", "detectTime", "pyramidLv", "qrcodeVersion", "srTime"]},
            "decodeScale": "F", "inBlackList": "Z", "inWhiteList": "Z",
        },
    }.items()]

    # Calling call_native directly avoids the upstream wrapper's global-name
    # lookup for dynamically constructed classes and retains local result refs.
    signatures = {
        "GetVersion": "()Ljava/lang/String;",
        "Init": "(IZZZZIILjava/lang/String;Ljava/lang/String;Lcom/tencent/qbar/QbarNative$QbarAiModelParam;)I",
        "SetReaders": "([III)I", "ScanImage": "([BIII)I",
        "GetResults": "([Lcom/tencent/qbar/QbarNative$QBarResultJNI;I)I",
        "Release": "(I)I",
    }
    namespace = {}
    for name, signature in signatures.items():
        def placeholder(*args):
            pass
        namespace[name] = staticmethod(java_method_def(name, signature, native=True)(placeholder))
    QbarNative = JavaClassDef("QbarNative", (), namespace, jvm_name="com/tencent/qbar/QbarNative")

    old_stdout = sys.stdout
    try:
        emu = Emulator(vfs_root=str(vfs_root),
                       config_path=str(runtime_root / "emu_cfg/default.json"),
                       arch=emu_const.ARCH_ARM64, muti_task=True)
    finally:
        sys.stdout = old_stdout
    try:
        for cls in [QbarNative, Model, Result, *extra_classes]:
            emu.java_classloader.add_class(cls)
        emu.system_properties.update({"ro.kernel.qemu": "0", "libc.debug.malloc": ""})

        @native_method
        def log_stub(mu, priority, tag, fmt):
            return 0

        @native_method
        def bitmap_stub(mu, env, bitmap, output):
            return (1 << 64) - 1

        @native_method
        def lifecycle_stub(mu):
            # No forks or dlclose in this disposable helper. The host process
            # owns memory cleanup; no emulated teardown callbacks need registration.
            return 0

        hooker = emu._Emulator__hooker
        for symbol in ["__android_log_print", "__android_log_vprint"]:
            emu.modules.add_symbol_hook(symbol, hooker.write_function(log_stub))
        for symbol in ["AndroidBitmap_getInfo", "AndroidBitmap_lockPixels", "AndroidBitmap_unlockPixels"]:
            emu.modules.add_symbol_hook(symbol, hooker.write_function(bitmap_stub))
        for symbol in ["__cxa_finalize", "__cxa_atexit", "__register_atfork"]:
            emu.modules.add_symbol_hook(symbol, hooker.write_function(lifecycle_stub))

        module = emu.load_library(str(runtime_root / "libwechatQrMod.so"))
        emu.call_symbol(module, "JNI_OnLoad", emu.java_vm.address_ptr, 0)
        env = emu.java_vm.jni_env

        def call(name, *args):
            method = QbarNative.find_method(name, signatures[name])
            if not method.native_addr:
                raise RuntimeError("JNI native not registered: " + name)
            clazz = env.add_local_reference(jclass(QbarNative.class_object))
            return emu.call_native(method.native_addr, env.address_ptr, clazz, *args)

        model = Model()
        model.detectModelVersion = String("V1.0.0.26")
        model.superResolutionModelVersion = String("V1.0.0.26")
        handle = call("Init", 1, 0, 0, 1, 0, 1, 1, String("ANY"), String("UTF-8"), model)
        if handle & 0x80000000:
            raise RuntimeError("Init failed: " + str(handle))
        try:
            status = call("SetReaders", Array([1, 2, 3, 4, 5]), 5, handle)
            if status:
                raise RuntimeError("SetReaders failed: " + str(status))
            status = call("ScanImage", ByteArray(bytearray(gray)), width, height, handle)
            if status:
                raise RuntimeError("ScanImage failed: " + str(status))
            results = [Result() for _ in range(3)]
            count = call("GetResults", Array(results), handle)
            if count & 0x80000000:
                raise RuntimeError("GetResults failed: " + str(count))
            if count == 0:
                return {"error": "NO_CODE"}
            if count != 1:
                return {"error": "MULTIPLE_CODES"}
            present = [result for result in results if result.typeName.get_py_string()]
            if len(present) > 1:
                return {"error": "MULTIPLE_CODES"}
            if not present:
                return {"error": "NO_CODE"}
            result = present[0]
            if result.typeName.get_py_string() != "WX_CODE":
                return {"error": "WRONG_CODE_TYPE"}
            payload = bytes(result.data.get_py_items())
            if not payload:
                return {"error": "NO_CODE"}
            return {"scanType": "WX_CODE", "rawBase64": base64.b64encode(payload).decode("ascii")}
        finally:
            call("Release", handle)

    finally:
        _close_emulator(emu)
