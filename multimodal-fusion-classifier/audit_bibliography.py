"""Audit the local BibTeX candidate bank through Pandoc's parser."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
EXPECTED_KEYS = (
    "white2020wastenet",
    "proenca2020taco",
    "bashkirova2022zerowaste",
    "bihler2023multisensor",
    "casao2024spectralwaste",
    "inacio2026sortwaste",
    "crall2026scatspotter",
    "ma2022missing",
    "zhang2023quality",
    "wang2023dicmor",
    "gao2024aleatoric",
    "chen2024pcd",
    "zhuang2025hme",
    "radford2021clip",
    "perez2018film",
    "lin2017focal",
    "kendall2017uncertainty",
    "bai2025qwen3vl",
    "guo2017calibration",
    "ultralytics2024yolo11",
)
DOMAIN_COVERAGE = {
    "waste_visual_benchmarks": (
        "white2020wastenet",
        "proenca2020taco",
        "bashkirova2022zerowaste",
        "inacio2026sortwaste",
        "crall2026scatspotter",
    ),
    "waste_physical_multisensor": (
        "bihler2023multisensor",
        "casao2024spectralwaste",
    ),
    "missing_or_low_quality_modalities": (
        "ma2022missing",
        "zhang2023quality",
        "wang2023dicmor",
        "gao2024aleatoric",
        "chen2024pcd",
        "zhuang2025hme",
    ),
    "method_foundations": (
        "radford2021clip",
        "perez2018film",
        "lin2017focal",
        "kendall2017uncertainty",
        "guo2017calibration",
        "ultralytics2024yolo11",
    ),
    "soft_observation_model": ("bai2025qwen3vl",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit BibTeX candidate structure using Pandoc"
    )
    parser.add_argument(
        "--bibliography",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "qwen-material"
        / "reference_candidates.bib",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "qwen-material"
        / "BIBLIOGRAPHY_AUDIT.json",
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def parse_with_pandoc(path: Path) -> tuple[list[dict[str, Any]], str]:
    version = subprocess.run(
        ["pandoc", "--version"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.splitlines()[0]
    completed = subprocess.run(
        [
            "pandoc",
            "-f",
            "biblatex",
            "-t",
            "csljson",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    value = json.loads(completed.stdout)
    if not isinstance(value, list) or not all(
        isinstance(item, dict) for item in value
    ):
        raise ValueError("Pandoc CSL JSON root must be a list of objects")
    return value, version


def audit_bibliography_items(
    items: Sequence[Mapping[str, Any]],
    *,
    raw_keys: Sequence[str],
) -> dict[str, Any]:
    errors: list[str] = []
    parsed_keys = [str(item.get("id", "")) for item in items]
    expected = set(EXPECTED_KEYS)
    if len(raw_keys) != len(set(raw_keys)):
        errors.append("raw BibTeX contains duplicate citation keys")
    if len(parsed_keys) != len(set(parsed_keys)):
        errors.append("Pandoc output contains duplicate citation keys")
    if set(raw_keys) != expected:
        errors.append("raw BibTeX citation key set mismatch")
    if set(parsed_keys) != expected:
        errors.append("Pandoc citation key set mismatch")

    doi_owners: dict[str, str] = {}
    url_owners: dict[str, str] = {}
    entries: list[dict[str, Any]] = []
    for item in items:
        key = str(item.get("id", ""))
        title = item.get("title")
        authors = item.get("author")
        issued = item.get("issued")
        url = item.get("URL")
        doi = item.get("DOI")
        if not isinstance(title, str) or not title.strip():
            errors.append(f"bibliography entry lacks title: {key}")
        if not isinstance(authors, list) or not authors:
            errors.append(f"bibliography entry lacks author: {key}")
        try:
            year = int(issued["date-parts"][0][0])
        except (KeyError, IndexError, TypeError, ValueError):
            errors.append(f"bibliography entry lacks valid year: {key}")
            year = None
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            errors.append(f"bibliography entry lacks valid URL: {key}")
        normalized_doi = (
            str(doi).strip().lower() if isinstance(doi, str) else ""
        )
        normalized_url = (
            str(url).strip().lower() if isinstance(url, str) else ""
        )
        if normalized_doi:
            if normalized_doi in doi_owners:
                errors.append(
                    "duplicate DOI: "
                    f"{normalized_doi}: {doi_owners[normalized_doi]}, {key}"
                )
            doi_owners[normalized_doi] = key
        if normalized_url:
            if normalized_url in url_owners:
                errors.append(
                    "duplicate URL: "
                    f"{normalized_url}: {url_owners[normalized_url]}, {key}"
                )
            url_owners[normalized_url] = key
        entries.append(
            {
                "id": key,
                "title": title,
                "year": year,
                "author_count": len(authors) if isinstance(authors, list) else 0,
                "doi": doi,
                "url": url,
            }
        )

    coverage = {
        name: {
            "keys": list(keys),
            "complete": set(keys).issubset(set(parsed_keys)),
        }
        for name, keys in DOMAIN_COVERAGE.items()
    }
    if not all(value["complete"] for value in coverage.values()):
        errors.append("bibliography domain coverage is incomplete")
    return {
        "valid": not errors,
        "error_count": len(errors),
        "errors": errors,
        "raw_entry_count": len(raw_keys),
        "parsed_entry_count": len(items),
        "citation_keys": sorted(parsed_keys),
        "domain_coverage": coverage,
        "entries": sorted(entries, key=lambda item: item["id"]),
    }


def audit_bibliography(path: Path) -> dict[str, Any]:
    path = _resolve(path)
    if not path.is_file():
        raise FileNotFoundError(f"bibliography is missing: {path}")
    text = path.read_text(encoding="utf-8-sig")
    raw_keys = re.findall(
        r"(?m)^@[A-Za-z]+\{\s*([^,\s]+)\s*,",
        text,
    )
    items, pandoc_version = parse_with_pandoc(path)
    result = audit_bibliography_items(items, raw_keys=raw_keys)
    return {
        "schema_version": 1,
        "state": "complete" if result["valid"] else "invalid",
        "source": {
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        },
        "parser": {
            "name": "pandoc",
            "version": pandoc_version,
            "input_format": "biblatex",
            "output_format": "csljson",
        },
        **result,
    }


def _write_markdown(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Bibliography Candidate Audit",
        "",
        f"- State: `{report['state']}`.",
        f"- Parsed entries: {report['parsed_entry_count']}.",
        f"- Unique citation keys: {len(set(report['citation_keys']))}.",
        (
            f"- Source: `{report['source']['path']}`, SHA-256 "
            f"`{report['source']['sha256']}`."
        ),
        (
            f"- Parser: `{report['parser']['version']}` "
            "(`biblatex` to `csljson`)."
        ),
        "",
        "## Coverage",
        "",
        "| Group | Entries | Complete |",
        "|---|---:|---:|",
    ]
    for name, coverage in report["domain_coverage"].items():
        lines.append(
            f"| `{name}` | {len(coverage['keys'])} | "
            f"{str(coverage['complete']).lower()} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "This audit verifies local structure, parser compatibility, "
                "key coverage and duplicate identifiers. It does not replace "
                "full-text reading, venue-fit review or online retraction checks."
            ),
            "",
        ]
    )
    if report["errors"]:
        lines.extend(
            [
                "## Errors",
                "",
                *[f"- {error}" for error in report["errors"]],
                "",
            ]
        )
    _atomic_write(path, "\n".join(lines))


def main() -> None:
    args = parse_args()
    report = audit_bibliography(args.bibliography)
    output = _resolve(args.output)
    _atomic_write(
        output,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    markdown = output.with_suffix(".md")
    _write_markdown(report, markdown)
    print(
        json.dumps(
            {
                "state": report["state"],
                "valid": report["valid"],
                "entry_count": report["parsed_entry_count"],
                "error_count": report["error_count"],
                "output": output.as_posix(),
                "markdown": markdown.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

