"""Default device-local WeChat circle decoder. No Java and no server decoding.

Native/code resources ship beside this file. On first use, missing Python wheels
are installed into a private cache, never into the device application's environment.
Use FTX_ROUNDCODE_AUTO_INSTALL=0 for managed/offline installations.
"""
from __future__ import annotations

import base64
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile

HERE = Path(__file__).resolve().parent
ASSETS = HERE / 'roundcode_assets'
MAX_IMAGE = 512 * 1024
PACKAGES = ('unicorn==2.1.4', 'capstone==5.0.6', 'Pillow>=10,<13')


def cache_root():
    configured = os.environ.get('FTX_ROUNDCODE_CACHE')
    base = Path(configured) if configured else Path(os.environ.get('LOCALAPPDATA')
        or os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache') / 'fentouxia/roundcode'
    return base.resolve()


def dependency_root():
    identity = hashlib.sha256(sys.executable.encode()).hexdigest()[:10]
    return cache_root() / f'python-{sys.version_info.major}.{sys.version_info.minor}-{platform.machine()}-{identity}'


def runtime_ready():
    """Cheap parent-process check to allow more time for first-use wheel setup."""
    target = dependency_root()
    return (target / '.ready').is_file() and (_system_dependencies_available()
        or all((target / name).is_dir() for name in ('unicorn', 'capstone', 'PIL')))


def _system_dependencies_available():
    try:
        return (importlib.metadata.version('unicorn') == '2.1.4'
            and importlib.metadata.version('capstone') == '5.0.6'
            and 10 <= int(importlib.metadata.version('Pillow').split('.')[0]) < 13)
    except (importlib.metadata.PackageNotFoundError, ValueError):
        return False


@contextlib.contextmanager
def setup_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - path.stat().st_mtime > 180:
                    path.unlink()
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() - started > 30:
                raise RuntimeError('SETUP_BUSY')
            time.sleep(.2)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def prepare_resources():
    archive = ASSETS / 'runtime.zip'
    expected = (ASSETS / 'runtime.sha256').read_text(encoding='ascii').strip()
    if len(expected) != 64 or hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
        raise RuntimeError('RUNTIME_INTEGRITY_FAILED')
    destination = cache_root() / ('runtime-' + expected[:16])
    with setup_lock(cache_root() / 'resources.lock'):
        if (destination / '.ready').is_file():
            return destination
        with tempfile.TemporaryDirectory(prefix='extract-', dir=cache_root()) as folder:
            staging = Path(folder) / 'runtime'
            staging.mkdir()
            with zipfile.ZipFile(archive) as zipped:
                entries = zipped.infolist()
                if len(entries) > 160 or sum(item.file_size for item in entries) > 24 * 1024 * 1024:
                    raise RuntimeError('RUNTIME_INTEGRITY_FAILED')
                for item in entries:
                    name = Path(item.filename)
                    if name.is_absolute() or '..' in name.parts or '\\' in item.filename or ':' in item.filename:
                        raise RuntimeError('RUNTIME_INTEGRITY_FAILED')
                zipped.extractall(staging)
            manifest = json.loads((staging / 'manifest.json').read_text(encoding='utf-8'))
            for name, digest in manifest['files'].items():
                path = (staging / name).resolve()
                if not path.is_relative_to(staging.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise RuntimeError('RUNTIME_INTEGRITY_FAILED')
            (staging / '.ready').write_text(expected, encoding='ascii')
            if destination.exists():
                # Only our exact content-addressed, private cache directory.
                shutil.rmtree(destination)
            staging.rename(destination)
    return destination


def prepare_dependencies():
    target = dependency_root()
    with setup_lock(cache_root() / (target.name + '.lock')):
        # Prefer an already compatible device environment; do not import it here.
        available = _system_dependencies_available()
        private_ready = (target / '.ready').is_file() and all((target / name).is_dir() for name in ('unicorn', 'capstone', 'PIL'))
        if not available and not private_ready:
            if os.environ.get('FTX_ROUNDCODE_AUTO_INSTALL', '1').lower() in ('0', 'false', 'no'):
                raise RuntimeError('DEPENDENCIES_MISSING')
            with tempfile.TemporaryDirectory(prefix='wheels-', dir=cache_root()) as folder:
                staged = Path(folder) / 'deps'
                result = subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                    '--no-input', '--only-binary=:all:', '--target', str(staged), *PACKAGES],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=90,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), shell=False)
                if result.returncode:
                    raise RuntimeError('DEPENDENCIES_MISSING')
                if target.exists():
                    shutil.rmtree(target)
                staged.rename(target)
        target.mkdir(exist_ok=True)
        if not available:
            sys.path.insert(0, str(target))
        (target / '.ready').touch()
    return target


def decode(image, mime):
    if not isinstance(image, bytes) or not 0 < len(image) <= MAX_IMAGE or mime not in ('image/png', 'image/jpeg'):
        return {'error': 'INVALID_IMAGE'}
    if not (image.startswith(b'\x89PNG\r\n\x1a\n') if mime == 'image/png' else image.startswith(b'\xff\xd8\xff')):
        return {'error': 'INVALID_IMAGE'}
    runtime = prepare_resources()
    prepare_dependencies()
    sys.path.insert(0, str(runtime))
    from roundcode_runtime_adapter import decode_bytes
    # The caller also bounds the process lifetime. This watchdog limits native
    # work after a potentially slower first-use dependency setup.
    watchdog = threading.Timer(15, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    try:
        with tempfile.TemporaryDirectory(prefix='decode-', dir=cache_root()) as folder:
            return decode_bytes(image, runtime, Path(folder))
    finally:
        watchdog.cancel()


def main():
    if sys.argv[1:] == ['--prepare']:
        prepare_resources()
        prepare_dependencies()
        print(json.dumps({'ready': True}))
        return
    if sys.argv[1:] != ['--stdin-json']:
        raise ValueError('Use --stdin-json or --prepare')
    data = sys.stdin.buffer.read(700000)
    if len(data) >= 700000:
        raise ValueError('Image too large')
    request = json.loads(data)
    encoded = request['imageBase64']
    if not isinstance(encoded, str) or len(encoded) > 699052:
        raise ValueError('Invalid image')
    image = base64.b64decode(encoded, validate=True)
    if base64.b64encode(image).decode('ascii') != encoded:
        raise ValueError('Invalid image')
    print(json.dumps(decode(image, request['mime'])))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Never emit image bytes, task tickets or opaque QR payloads in errors.
        reason = str(error) if isinstance(error, RuntimeError) and str(error) in {
            'SETUP_BUSY', 'RUNTIME_INTEGRITY_FAILED', 'DEPENDENCIES_MISSING'} else 'DECODER_UNAVAILABLE'
        print(json.dumps({'error': reason}))
        raise SystemExit(1)
