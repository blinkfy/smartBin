import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_integrated_protocol_binds_current_yolo_queue_source():
    protocol = json.loads(
        (
            PROJECT_ROOT
            / "reports"
            / "qwen-material"
            / "yolo-seg"
            / "INTEGRATED_EVALUATION_PROTOCOL.json"
        ).read_text(encoding="utf-8")
    )
    binding = protocol["inputs"]["queue_source"]
    source = PROJECT_ROOT / binding["path"]

    assert source.is_file()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == binding["sha256"]
