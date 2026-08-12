from __future__ import annotations

import copy

from audit_bibliography import EXPECTED_KEYS, audit_bibliography_items


def _items():
    return [
        {
            "id": key,
            "title": f"Title {index}",
            "author": [{"literal": f"Author {index}"}],
            "issued": {"date-parts": [[2020 + index % 6]]},
            "URL": f"https://example.test/{key}",
            "DOI": f"10.1234/{key}",
        }
        for index, key in enumerate(EXPECTED_KEYS)
    ]


def test_bibliography_audit_accepts_expected_complete_bank():
    result = audit_bibliography_items(
        _items(),
        raw_keys=list(EXPECTED_KEYS),
    )

    assert result["valid"] is True
    assert result["parsed_entry_count"] == 20
    assert all(
        row["complete"] for row in result["domain_coverage"].values()
    )


def test_bibliography_audit_rejects_missing_title_and_duplicate_doi():
    items = _items()
    items[0]["title"] = ""
    items[1]["DOI"] = items[0]["DOI"]

    result = audit_bibliography_items(
        items,
        raw_keys=list(EXPECTED_KEYS),
    )

    assert result["valid"] is False
    assert any("lacks title" in error for error in result["errors"])
    assert any("duplicate DOI" in error for error in result["errors"])


def test_bibliography_audit_rejects_key_drift():
    items = copy.deepcopy(_items())
    items[-1]["id"] = "unexpected"

    result = audit_bibliography_items(
        items,
        raw_keys=list(EXPECTED_KEYS),
    )

    assert result["valid"] is False
    assert any("citation key set mismatch" in error for error in result["errors"])

