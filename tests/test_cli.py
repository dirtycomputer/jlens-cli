from __future__ import annotations

import json
from pathlib import Path

import pytest

from jlens_cli.cli import main
from jlens_cli.io import CLIError, indices, read_records
from jlens_cli.report import render


@pytest.mark.parametrize(
    ("spec", "size", "expected"),
    [
        ("-1", 4, [3]),
        ("0:4:2", 4, [0, 2]),
        ("::-1", 3, [2, 1, 0]),
        ("0,-1", 4, [0, 3]),
        ("all", 2, [0, 1]),
    ],
)
def test_indices(spec, size, expected):
    assert indices(spec, size) == expected


@pytest.mark.parametrize("spec", ["", "7", "-6", "0,0", "2:2", "1:3:0", "a", "1:2:3:4"])
def test_invalid_indices(spec):
    with pytest.raises(CLIError):
        indices(spec, 4)


def test_jsonl_errors_and_field_selection(tmp_path):
    path = tmp_path / "input.jsonl"
    path.write_text('{"id":"a","prompt":"hello"}\n\n{"prompt":"world"}\n')
    assert read_records(str(path), text_field="prompt") == [
        {"id": "a", "text": "hello"},
        {"id": 1, "text": "world"},
    ]
    with pytest.raises(CLIError, match="non-empty"):
        read_records(str(path))
    path.write_text("{}\nnot-json\n")
    with pytest.raises(CLIError, match="line 2"):
        read_records(str(path))


def test_errors_are_json_on_stderr(capsys):
    assert main(["apply"]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err)["error"]["code"] == "usage_error"


def test_schema_and_help_are_dependency_free(run_cli):
    response = json.loads(run_cli("schema").stdout)
    assert response["schema_version"] == "1.0"
    assert "apply" in response["data"]["commands"]
    assert "--positions" in run_cli("apply", "--help").stdout


@pytest.mark.parametrize(
    "extra",
    [
        ["--dim-batch", "0"],
        ["--skip-first", "-1"],
        ["--num-shards", "2", "--shard-index", "2"],
        ["--max-seq-len", "16", "--skip-first", "16"],
    ],
)
def test_invalid_fit_fails_before_model_load(extra, capsys):
    assert (
        main(
            ["fit", "--model", "never-download", "--prompts", "missing.txt", "-o", "out.pt", *extra]
        )
        == 2
    )
    assert json.loads(capsys.readouterr().err)["status"] == "error"


def test_output_must_not_overwrite_input(capsys, tmp_path):
    path = tmp_path / "source.txt"
    path.write_text("keep me")
    assert (
        main(["fit", "--model", "no-model", "--prompts", str(path), "-o", str(path), "--force"])
        == 2
    )
    assert "overwrite input" in json.loads(capsys.readouterr().err)["error"]["message"]
    assert path.read_text() == "keep me"


def test_html_escapes_untrusted_content():
    report = render({"prompt": "</script><img src=x onerror=alert(1)>"}, title="<bad>")
    assert "<title>&lt;bad&gt;</title>" in report
    assert "</script><img" not in report
    assert "\\u003c/script>" in report
    assert "<script src=" not in report
    assert "textContent" in report


def test_no_model_packages_imported_by_help():
    # Importing the package/CLI should stay cheap for agent discovery.
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import jlens_cli.cli,sys; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_text_and_json_input(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text('["first", {"id":"中文","text":"第二个"}]')
    assert read_records(str(path))[1] == {"id": "中文", "text": "第二个"}
    path.write_text('{"text":"not an array"}')
    with pytest.raises(CLIError, match="array"):
        read_records(str(path))
    path = Path(tmp_path / "prompts.txt")
    path.write_text("hello\n\n world \n")
    assert [row["text"] for row in read_records(str(path))] == ["hello", " world "]
