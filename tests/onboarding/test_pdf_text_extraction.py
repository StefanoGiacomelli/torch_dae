from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


def script(repo_root: Path) -> Path:
    return repo_root / "skills/audio-model-onboarding/scripts/extract_pdf_text.py"


def write_pdf(
    path: Path,
    *,
    texts: tuple[str, ...] = (),
    blank_pages: int = 0,
    encrypted: bool = False,
) -> None:
    writer = PdfWriter()
    for text in texts:
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        font_reference = writer._add_object(font)
        resources = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_reference})}
        )
        content = DecodedStreamObject()
        content.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Resources")] = resources
        page[NameObject("/Contents")] = writer._add_object(content)
    for _ in range(blank_pages):
        writer.add_blank_page(width=612, height=792)
    writer.add_metadata({"/Title": "Synthetic PDF"})
    if encrypted:
        writer.encrypt("synthetic-password")
    with path.open("wb") as stream:
        writer.write(stream)


def run_json(
    repo_root: Path,
    path: Path,
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script(repo_root)), str(path), "--json", *arguments],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )


def test_minimal_valid_pdf_and_no_text_layer(repo_root: Path, tmp_path: Path) -> None:
    minimal = tmp_path / "minimal.pdf"
    no_text = tmp_path / "no-text.pdf"
    write_pdf(minimal)
    write_pdf(no_text, blank_pages=1)

    minimal_result = run_json(repo_root, minimal)
    no_text_result = run_json(repo_root, no_text)

    assert minimal_result.returncode == 0, minimal_result.stderr
    assert no_text_result.returncode == 0, no_text_result.stderr
    assert json.loads(minimal_result.stdout)["status"] == "no_text_layer"
    payload = json.loads(no_text_result.stdout)
    assert payload["status"] == "no_text_layer"
    assert payload["has_text_layer"] is False
    assert payload["ocr_provided"] is False
    assert payload["page_count"] == 1


def test_extractable_text_metadata_boundaries_and_determinism(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    path = tmp_path / "text.pdf"
    write_pdf(path, texts=("First page", "Second page"))

    first = run_json(repo_root, path)
    second = run_json(repo_root, path)
    plain = subprocess.run(
        [sys.executable, str(script(repo_root)), str(path)],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert first.returncode == second.returncode == plain.returncode == 0
    assert first.stdout == second.stdout
    payload = json.loads(first.stdout)
    assert payload["status"] == "ok"
    assert payload["metadata"]["/Title"] == "Synthetic PDF"
    assert [page["text"].strip() for page in payload["pages"]] == [
        "First page",
        "Second page",
    ]
    assert "--- page 1 ---" in plain.stdout
    assert "--- page 2 ---" in plain.stdout
    assert "OCR: not provided" in plain.stdout


@pytest.mark.parametrize(
    ("fixture_name", "content", "expected_status"),
    [
        ("malformed.pdf", b"%PDF-1.7\nnot a valid PDF", "parser_failure"),
        ("not-pdf.txt", b"ordinary text", "not_pdf"),
    ],
)
def test_malformed_and_non_pdf_inputs(
    repo_root: Path,
    tmp_path: Path,
    fixture_name: str,
    content: bytes,
    expected_status: str,
) -> None:
    path = tmp_path / fixture_name
    path.write_bytes(content)
    result = run_json(repo_root, path)
    assert result.returncode == 2
    assert json.loads(result.stdout)["status"] == expected_status
    assert not result.stderr


def test_encrypted_pdf_is_explicit(repo_root: Path, tmp_path: Path) -> None:
    path = tmp_path / "encrypted.pdf"
    write_pdf(path, texts=("Secret",), encrypted=True)
    result = run_json(repo_root, path)
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "encrypted"
    assert payload["ocr_provided"] is False


def test_file_page_character_and_override_limits(repo_root: Path, tmp_path: Path) -> None:
    path = tmp_path / "limited.pdf"
    write_pdf(path, texts=("A long synthetic text layer", "second"))

    file_result = run_json(repo_root, path, "--max-file-size-bytes", "10")
    page_result = run_json(repo_root, path, "--max-pages", "1")
    character_result = run_json(repo_root, path, "--max-characters", "5")
    invalid_override = run_json(
        repo_root,
        path,
        "--max-pages",
        "1001",
    )

    for result, limit_name in (
        (file_result, "file_size_bytes"),
        (page_result, "page_count"),
        (character_result, "extracted_characters"),
    ):
        assert result.returncode == 2
        payload = json.loads(result.stdout)
        assert payload["status"] == "limit_exceeded"
        assert payload["limit"]["name"] == limit_name
    assert invalid_override.returncode == 2
    assert json.loads(invalid_override.stdout)["status"] == "invalid_limit"


def test_extraction_does_not_mutate_repository(repo_root: Path, tmp_path: Path) -> None:
    path = tmp_path / "text.pdf"
    write_pdf(path, texts=("Read only",))
    before = subprocess.run(
        ["git", "status", "--porcelain=v1", "-uall"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    result = run_json(repo_root, path)
    after = subprocess.run(
        ["git", "status", "--porcelain=v1", "-uall"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert result.returncode == 0
    assert after == before
