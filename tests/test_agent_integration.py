from __future__ import annotations

import hashlib
import json

import pytest

pytestmark = pytest.mark.integration


def fit_args(model, corpus, output):
    return [
        "fit",
        "--model",
        model,
        "--prompts",
        corpus,
        "-o",
        output,
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--max-seq-len",
        "32",
        "--skip-first",
        "0",
        "--dim-batch",
        "4",
        "--save-dtype",
        "float32",
        "--local-files-only",
    ]


def test_real_transformer_full_workflow(tiny_model, corpus, tmp_path, run_cli):
    import torch
    from jlens import JacobianLens

    weight = tiny_model / "model.safetensors"
    before = hashlib.sha256(weight.read_bytes()).hexdigest()
    output = tmp_path / "lens.pt"
    checkpoint = tmp_path / "checkpoint.pt"
    args = fit_args(tiny_model, corpus, output) + ["--checkpoint", checkpoint]
    fitted = json.loads(run_cli(*args).stdout)["data"]
    assert fitted["n_prompts"] == 2
    assert fitted["source_layers"] == [0, 1]
    assert checkpoint.exists()
    assert json.loads(run_cli("inspect", "--lens", output).stdout)["data"]["d_model"] == 8

    base = [
        "--model",
        tiny_model,
        "--lens",
        output,
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--local-files-only",
    ]
    prediction = json.loads(
        run_cli(
            "apply",
            *base,
            "--prompt",
            "the cat sat on a mat",
            "--positions=-2,-1",
            "--top-k",
            "5",
            "--baseline",
            "--track-token-id",
            "3",
        ).stdout
    )["data"]
    sample = prediction["results"][0]
    assert sample["positions"] == [sample["num_tokens"] - 2, sample["num_tokens"] - 1]
    assert [r["kind"] for r in sample["readouts"]] == [
        "jacobian",
        "jacobian",
        "logit",
        "logit",
        "model",
    ]
    for row in sample["readouts"]:
        for cell in row["positions"]:
            probs = [token["probability"] for token in cell["top_tokens"]]
            assert probs == sorted(probs, reverse=True)
            assert 0 < sum(probs) < 1
            assert cell["tracked_tokens"][0]["rank"] >= 1

    # Verify the CLI numbers against direct upstream output (not a mocked adapter).
    import jlens
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = jlens.from_hf(
        AutoModelForCausalLM.from_pretrained(
            tiny_model, dtype=torch.float32, attn_implementation="eager"
        ),
        AutoTokenizer.from_pretrained(tiny_model),
    )
    lens = JacobianLens.load(str(output))
    direct, _, _ = lens.apply(model, sample["prompt"], positions=sample["positions"])
    expected_probs, expected_ids = direct[0][0].softmax(-1).topk(5)
    actual = sample["readouts"][0]["positions"][0]["top_tokens"]
    assert [item["token_id"] for item in actual] == expected_ids.tolist()
    torch.testing.assert_close(
        torch.tensor([item["probability"] for item in actual]), expected_probs
    )

    batch = run_cli("apply", *base, "--input", corpus, "--output-format", "jsonl")
    records = [json.loads(line) for line in batch.stdout.splitlines()]
    assert [record["data"]["result"]["id"] for record in records] == [0, 1]
    result_file = tmp_path / "result.json"
    summary = json.loads(
        run_cli("apply", *base, "--prompt-file", "-", "-o", result_file, stdin="hello world").stdout
    )["data"]
    assert summary["num_prompts"] == 1
    assert json.loads(result_file.read_text())["data"]["results"][0]["prompt"] == "hello world"

    report = tmp_path / "report.html"
    run_cli("visualize", *base, "--prompt", "hello world", "-o", report, "--baseline")
    assert "application/json" in report.read_text()
    assert "<script src=" not in report.read_text()
    run_cli("visualize", *base, "--prompt", "hello world", "-o", report, expect=2)
    run_cli("apply", *base, "--prompt", "hello world", "--positions", "99", expect=2)
    run_cli("apply", *base, "--prompt", "hello world", "--layers", "2", expect=2)

    original = {key: value.clone() for key, value in lens.jacobians.items()}
    run_cli(*args, "--force")
    resumed = JacobianLens.load(str(output))
    for key in original:
        torch.testing.assert_close(original[key], resumed.jacobians[key], rtol=0, atol=0)
    mismatch = run_cli(*args, "--force", "--skip-first", "1", expect=2)
    assert json.loads(mismatch.stderr.splitlines()[-1])["error"]["code"] == "checkpoint_mismatch"
    run_cli(*args, "--force", "--no-resume")
    assert hashlib.sha256(weight.read_bytes()).hexdigest() == before


def test_shards_merge_like_full_fit(tiny_model, corpus, tmp_path, run_cli):
    import torch
    from jlens import JacobianLens

    full = tmp_path / "full.pt"
    run_cli(*fit_args(tiny_model, corpus, full))
    paths = [tmp_path / f"shard-{index}.pt" for index in range(2)]
    for index, path in enumerate(paths):
        run_cli(*fit_args(tiny_model, corpus, path), "--num-shards", "2", "--shard-index", index)
    merged = tmp_path / "merged.pt"
    result = json.loads(run_cli("merge", *paths, "-o", merged, "--save-dtype", "float32").stdout)[
        "data"
    ]
    assert result["n_prompts"] == 2
    a, b = JacobianLens.load(str(full)), JacobianLens.load(str(merged))
    for layer in a.source_layers:
        torch.testing.assert_close(a.jacobians[layer], b.jacobians[layer])
    run_cli("merge", paths[0], paths[0], "-o", tmp_path / "bad.pt", expect=2)


def test_all_short_prompts_do_not_create_lens(tiny_model, tmp_path, run_cli):
    corpus = tmp_path / "short.txt"
    corpus.write_text("hello\nworld\n")
    output = tmp_path / "no-lens.pt"
    response = run_cli(*fit_args(tiny_model, corpus, output), "--skip-first", "4", expect=2)
    assert "no prompts were long enough" in response.stderr
    assert not output.exists()


def test_incompatible_and_nonfinite_lenses(tiny_model, tmp_path, run_cli):
    import torch
    from jlens import JacobianLens

    bad = tmp_path / "bad.pt"
    JacobianLens({0: torch.eye(9)}, n_prompts=1, d_model=9).save(str(bad))
    result = run_cli(
        "apply", "--model", tiny_model, "--lens", bad, "--prompt", "hello world", expect=2
    )
    assert "incompatible_lens" in result.stderr
    JacobianLens({0: torch.full((8, 8), float("nan"))}, n_prompts=1, d_model=8).save(str(bad))
    result = run_cli("inspect", "--lens", bad, expect=2)
    assert "NaN" in result.stderr
