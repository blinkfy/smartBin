import hashlib
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_runtime_recovery_redirects_only_hash_locked_source(tmp_path):
    source = tmp_path / "source.png"
    fallback = tmp_path / "fallback.png"
    untouched = tmp_path / "untouched.png"
    Image.new("RGB", (3, 2), (255, 0, 0)).save(source)
    Image.new("RGB", (3, 2), (0, 255, 0)).save(fallback)
    Image.new("RGB", (3, 2), (0, 0, 255)).save(untouched)
    environment = os.environ.copy()
    recovery_path = str(PROJECT_ROOT / "runtime_recovery")
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        recovery_path
        if not existing_pythonpath
        else recovery_path + os.pathsep + existing_pythonpath
    )
    environment.update(
        {
            "MMFC_IMAGE_RECOVERY_SOURCE": str(source),
            "MMFC_IMAGE_RECOVERY_FALLBACK": str(fallback),
            "MMFC_IMAGE_RECOVERY_SOURCE_SHA256": _sha256(source),
            "MMFC_IMAGE_RECOVERY_FALLBACK_SHA256": _sha256(fallback),
        }
    )
    script = (
        "from pathlib import Path;"
        "from PIL import Image;"
        f"a=Image.open(Path({str(source)!r})).getpixel((0,0));"
        f"b=Image.open(Path({str(untouched)!r})).getpixel((0,0));"
        "print(a,b)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0
    assert result.stdout.strip().endswith("(0, 255, 0) (0, 0, 255)")
    assert "[mmfc-image-recovery] active:" in result.stdout


def test_runtime_recovery_rejects_hash_drift(tmp_path):
    source = tmp_path / "source.png"
    fallback = tmp_path / "fallback.png"
    Image.new("RGB", (2, 2), (255, 0, 0)).save(source)
    Image.new("RGB", (2, 2), (0, 255, 0)).save(fallback)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "runtime_recovery")
    environment.update(
        {
            "MMFC_IMAGE_RECOVERY_SOURCE": str(source),
            "MMFC_IMAGE_RECOVERY_FALLBACK": str(fallback),
            "MMFC_IMAGE_RECOVERY_SOURCE_SHA256": "0" * 64,
            "MMFC_IMAGE_RECOVERY_FALLBACK_SHA256": _sha256(fallback),
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", "print('should not run')"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode != 0
    assert "source SHA-256 mismatch" in result.stderr
