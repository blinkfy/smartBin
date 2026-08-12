from __future__ import annotations

import pytest

from build_manuscript_result_governance_v8 import _extract_title


def test_extract_option_a_title():
    value = _extract_title(
        r"\title{\textbf{面向图像条件软传感器退化的可靠多模态垃圾分类方法}}"
    )

    assert value == "面向图像条件软传感器退化的可靠多模态垃圾分类方法"


def test_extract_title_rejects_unstructured_title():
    with pytest.raises(ValueError, match="could not be extracted"):
        _extract_title(r"\title{plain}")
