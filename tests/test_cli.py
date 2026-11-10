import json
import subprocess
import sys

import pytest

from jlens_cli.legacy import main, read_corpus


def test_help_is_lightweight():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from jlens_cli.legacy import main; "
            "main(['presets']); assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["qwen3.5-4b"]["model"] == "Qwen/Qwen3.5-4B"


def test_bad_arguments(capsys):
    with pytest.raises(SystemExit) as error:
        main(["apply", "--prompt", "hello", "--top-k", "0"])
    assert error.value.code == 2
    assert "positive integer" in capsys.readouterr().err


def test_empty_prompt(capsys):
    assert main(["apply", "--prompt", " "]) == 1
    assert "empty" in capsys.readouterr().err


def test_corpus_validation(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text('{"text":"hello"}\n{"text":"world"}\n')
    assert read_corpus(path, limit=1) == ["hello"]
    path.write_text('{"text":17}\n')
    with pytest.raises(ValueError, match=":1:"):
        read_corpus(path)


def test_existing_output_is_protected(tmp_path, capsys):
    path = tmp_path / "output.json"
    path.write_text("keep me")
    assert main(["inspect", "--lens", "missing.pt", "-o", str(path)]) == 1
    assert path.read_text() == "keep me"
    assert "--force" in capsys.readouterr().err
