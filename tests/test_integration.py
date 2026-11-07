"""Exercise actual upstream fitting and inference with a tiny local Llama.

No model downloads, API keys, CUDA, or mocked numerical engine.
"""

import json

import pytest

torch = pytest.importorskip("torch")
jlens = pytest.importorskip("jlens")
transformers = pytest.importorskip("transformers")

from jlens_cli.cli import main  # noqa: E402


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    torch.set_num_threads(1)
    torch.manual_seed(7)
    path = tmp_path_factory.mktemp("model")
    vocabulary = {
        token: i
        for i, token in enumerate(
            ["[UNK]", "[BOS]", "[EOS]", "the", "cat", "sat", "on", "a", "mat", "."]
        )
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", bos_token="[BOS]", eos_token="[EOS]"
    )
    tokenizer.save_pretrained(path)
    config = transformers.LlamaConfig(
        vocab_size=len(vocabulary),
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
    )
    transformers.LlamaForCausalLM(config).save_pretrained(path)
    return path


@pytest.fixture
def fitted(tmp_path, tiny_model):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("the cat sat on a mat .\nthe cat sat .\n")
    output = tmp_path / "lens.pt"
    checkpoint = tmp_path / "checkpoint.pt"
    args = [
        "fit",
        "--model",
        str(tiny_model),
        "--device",
        "cpu",
        "--local-files-only",
        "--prompts",
        str(corpus),
        "--skip-first",
        "0",
        "--max-seq-len",
        "16",
        "--dim-batch",
        "4",
        "--checkpoint",
        str(checkpoint),
        "-o",
        str(output),
    ]
    assert main(args) == 0
    return output, args, corpus


def test_fit_resume_and_corpus_guard(fitted, capsys):
    output, args, corpus = fitted
    before = jlens.JacobianLens.load(str(output))
    assert before.n_prompts == 2
    assert before.source_layers == [0]
    assert main([*args, "--resume", "--force"]) == 0
    after = jlens.JacobianLens.load(str(output))
    torch.testing.assert_close(before.jacobians[0], after.jacobians[0])
    corpus.write_text("a different corpus now\n")
    assert main([*args, "--resume", "--force"]) == 1
    assert "metadata does not match" in capsys.readouterr().err


def test_apply_inspect_and_merge(fitted, tiny_model, tmp_path, capsys):
    output, _, _ = fitted
    capsys.readouterr()
    assert main(["inspect", "--lens", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["d_model"] == 8
    args = [
        "apply",
        "--model",
        str(tiny_model),
        "--lens",
        str(output),
        "--device",
        "cpu",
        "--local-files-only",
        "--prompt",
        "the cat sat on a mat .",
        "--positions=-2,-1",
        "--top-k",
        "3",
    ]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    # Upstream may enable BOS; positions index the encoded sequence, not words.
    sequence_length = len(result["input_token_ids"])
    assert result["positions"] == [sequence_length - 2, sequence_length - 1]
    assert len(result["layers"]["0"]) == 2
    assert len(result["layers"]["0"][0]) == 3
    assert result["input_tokens"] == ["mat", "."]
    assert all(0 <= item["probability"] <= 1 for item in result["model_output"][0])
    assert main([*args, "--logit-lens", "--format", "text"]) == 0
    assert f"Position {sequence_length - 2}" in capsys.readouterr().out
    assert main([*args, "--positions=-99"]) == 1
    assert "positions must" in capsys.readouterr().err
    # Unequal prompt weights verify the numerical merge contract.
    second = tmp_path / "second.pt"
    jlens.JacobianLens({0: torch.eye(8)}, n_prompts=6, d_model=8).save(str(second))
    merged = tmp_path / "merged.pt"
    assert main(["merge", str(output), str(second), "-o", str(merged)]) == 0
    lens = jlens.JacobianLens.load(str(merged))
    original = jlens.JacobianLens.load(str(output))
    assert lens.n_prompts == 8
    torch.testing.assert_close(
        lens.jacobians[0], original.jacobians[0] * 0.25 + torch.eye(8) * 0.75
    )


def test_render(fitted, tiny_model, tmp_path):
    output, _, _ = fitted
    html = tmp_path / "slice.html"
    assert (
        main(
            [
                "render",
                "--model",
                str(tiny_model),
                "--lens",
                str(output),
                "--device",
                "cpu",
                "--local-files-only",
                "--prompt",
                "the cat sat on a mat .",
                "--top-k",
                "3",
                "--title",
                "<test>",
                "-o",
                str(html),
            ]
        )
        == 0
    )
    page = html.read_text()
    assert "&lt;test&gt;" in page
    assert '"mode": "embed"' in page
    assert "__BOOTSTRAP__" not in page
    assert "<script src=" not in page


def test_incompatible_lens(tiny_model, tmp_path, capsys):
    path = tmp_path / "bad.pt"
    jlens.JacobianLens({0: torch.eye(4)}, n_prompts=1, d_model=4).save(str(path))
    assert (
        main(
            [
                "apply",
                "--model",
                str(tiny_model),
                "--lens",
                str(path),
                "--device",
                "cpu",
                "--prompt",
                "the cat",
            ]
        )
        == 1
    )
    assert "do not match" in capsys.readouterr().err
