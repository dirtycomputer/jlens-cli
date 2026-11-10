"""Numerical checkpoint compatibility and CLI boundary checks, entirely offline."""

import json

import pytest
from test_integration import tiny_model  # noqa: F401

from jlens_cli import jpp
from jlens_cli import legacy as cli

torch = pytest.importorskip("torch")


@pytest.fixture
def checkpoint(tmp_path, tiny_model):  # noqa: F811
    torch.manual_seed(23)
    path = tmp_path / "jpp.pt"
    torch.save(
        {
            "parameters": {"jacobians": {0: torch.randn(8, 8)}},
            "source_layers": [0],
            "config": {
                "hf_model_name": str(tiny_model),
                "d_model": 8,
                "num_prompts_trained_on": 64,
                "lens_type": "jacobian",
                "relative_end_transport_layer": -1,
            },
        },
        path,
    )
    return path


def test_numerical_readout(checkpoint, tiny_model, capsys):  # noqa: F811
    args = jpp.parser().parse_args(
        [
            "apply",
            "--model",
            str(tiny_model),
            "--lens",
            str(checkpoint),
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--local-files-only",
            "--prompt",
            "the cat sat .",
            "--top-k",
            "10",
        ]
    )
    jpp.resolve(args)
    model = cli.load_model(args)
    lens, config = jpp.load_checkpoint(checkpoint)
    captured = {}

    def hook(_module, _args, output):
        captured[0] = output[0] if isinstance(output, tuple) else output

    handle = model.layers[0].register_forward_hook(hook)
    with torch.no_grad():
        model.forward(model.encode(args.prompt))
        manual = model.unembed(captured[0][0, -1:].float() @ lens.jacobians[0].T)
    handle.remove()
    result = jpp.infer(args, model, lens, config, args.prompt)
    probabilities = manual.float().softmax(-1)[0]
    for item in result["layers"]["0"][0]:
        assert item["probability"] == pytest.approx(probabilities[item["token_id"]].item())
        assert any(c.isalnum() for c in item["token"])
    assert len(result["layers"]["0"][0]) == 9  # punctuation excluded
    assert jpp.legacy_main(["inspect", "--lens", str(checkpoint)]) == 0
    assert json.loads(capsys.readouterr().out)["d_model"] == 8
    assert (
        jpp.legacy_main(
            [
                "apply",
                "--model",
                str(tiny_model),
                "--lens",
                str(checkpoint),
                "--device",
                "cpu",
                "--local-files-only",
                "--prompt",
                "the cat",
                "--format",
                "text",
            ]
        )
        == 0
    )
    assert "Position" in capsys.readouterr().out


@pytest.mark.parametrize("mutation", ["target", "shape", "finite", "sources"])
def test_bad_checkpoint(checkpoint, mutation):
    obj = torch.load(checkpoint, weights_only=True)
    if mutation == "target":
        obj["config"]["relative_end_transport_layer"] = -2
    elif mutation == "shape":
        obj["parameters"]["jacobians"][0] = torch.eye(4)
    elif mutation == "finite":
        obj["parameters"]["jacobians"][0][0, 0] = float("nan")
    else:
        obj["source_layers"] = [1]
    torch.save(obj, checkpoint)
    with pytest.raises(ValueError):
        jpp.load_checkpoint(checkpoint)


def test_filter_probabilities_and_identity(checkpoint, tiny_model):  # noqa: F811
    class Tokens:
        def decode(self, ids):
            return ["!", "heart", "心脏", "4"][ids[0]]

    logits = torch.tensor([[8.0, 3.0, 2.0, 1.0]])
    mask = jpp.readout_mask(Tokens(), 4)
    result = jpp.top_tokens(logits, Tokens(), 10, mask)[0]
    assert [x["token_id"] for x in result] == [1, 2, 3]
    assert sum(x["probability"] for x in result) < 0.1
    lens, config = jpp.load_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="identity"):
        jpp.validate_model(lens, config, None, "wrong/model")
    assert (
        jpp.legacy_main(
            ["inspect", "--lens", str(checkpoint), "--output", str(checkpoint), "--force"]
        )
        == 1
    )


def test_unified_batch_readout(checkpoint, tiny_model, tmp_path, monkeypatch, capsys):  # noqa: F811
    from jlens_cli import operations

    corpus = tmp_path / "batch.jsonl"
    corpus.write_text('{"id":"a","text":"the cat sat ."}\n{"id":"b","text":"the cat"}\n')
    original_load = operations.load_model
    loads = []

    def load(args):
        model = original_load(args)
        loads.append(model)
        return model

    monkeypatch.setattr(operations, "load_model", load)
    base = [
        "--model",
        str(tiny_model),
        "--lens",
        str(checkpoint),
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--local-files-only",
    ]
    assert (
        jpp.main(
            [
                "apply",
                *base,
                "--input",
                str(corpus),
                "--output-format",
                "jsonl",
                "--track-token-id",
                "3",
                "--track-token-id",
                "9",
                "--top-k",
                "1",
                "--baseline",
            ]
        )
        == 0
    )
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(loads) == 1
    assert [r["data"]["result"]["id"] for r in records] == ["a", "b"]
    lens, config = jpp.load_checkpoint(checkpoint)
    for record in records:
        data = record["data"]
        assert data["backend"] == "jpp" and data["readout_filter"]
        assert data["probability_basis"] == "full-vocabulary"
        sample = data["result"]
        assert [r["kind"] for r in sample["readouts"]] == ["jpp", "logit", "model"]
        ids = [t["token_id"] for t in sample["tokens"]]
        logits, _, _ = lens.apply(loads[0], sample["prompt"], positions=sample["positions"])
        scores = logits[0][0]
        cell = sample["readouts"][0]["positions"][0]
        mask = jpp.readout_mask(loads[0].tokenizer, len(scores))
        for tracked in cell["tracked_tokens"]:
            token_id = tracked["token_id"]
            assert tracked["rank"] == int((scores > scores[token_id]).sum()) + 1
            assert tracked["probability"] == pytest.approx(scores.softmax(-1)[token_id].item())
            assert tracked["logit"] == pytest.approx(scores[token_id].item())
            if token_id == 9:  # Punctuation is excluded from J++ ranking, but remains trackable.
                assert tracked["filtered_rank"] is None
            else:
                assert (
                    tracked["filtered_rank"] == int(((scores > scores[token_id]) & mask).sum()) + 1
                )
        assert cell["input_token_id"] == ids[sample["positions"][0]]
        args = jpp.parser().parse_args(
            ["apply", *base, "--prompt", sample["prompt"], "--top-k", "1"]
        )
        jpp.resolve(args)
        legacy = jpp.infer(args, loads[0], lens, config, sample["prompt"])
        top = cell["top_tokens"][0]
        assert top["token_id"] == legacy["layers"]["0"][0][0]["token_id"]
        assert top["probability"] == pytest.approx(legacy["layers"]["0"][0][0]["probability"])

    # A late failure must leave stdout empty and the prior result file untouched.
    target = tmp_path / "prior.jsonl"
    target.write_text("keep me")
    assert (
        jpp.main(
            [
                "apply",
                *base,
                "--input",
                str(corpus),
                "--positions",
                "4",
                "--output",
                str(target),
                "--force",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err.splitlines()[-1])["error"]["code"] == "invalid_input"
    assert target.read_text() == "keep me"


def test_unified_artifacts_and_legacy(checkpoint, tiny_model, tmp_path, capsys):  # noqa: F811
    assert jpp.main(["inspect", "--lens", str(checkpoint)]) == 0
    assert json.loads(capsys.readouterr().out)["data"]["d_model"] == 8
    assert jpp.main(["download", "--lens", str(checkpoint), "--local-files-only"]) == 0
    assert json.loads(capsys.readouterr().out)["data"]["lens_path"] == str(checkpoint)
    assert jpp.main(["--legacy", "inspect", "--lens", str(checkpoint)]) == 0
    assert json.loads(capsys.readouterr().out)["d_model"] == 8
    html = tmp_path / "jpp.html"
    legacy_output = tmp_path / "legacy.json"
    assert (
        jpp.main(
            [
                "--legacy",
                "apply",
                "--model",
                str(tiny_model),
                "--lens",
                str(checkpoint),
                "--device",
                "cpu",
                "--local-files-only",
                "--prompt",
                "the cat",
                "--output",
                str(legacy_output),
            ]
        )
        == 0
    )
    assert "layers" in json.loads(legacy_output.read_text())
    capsys.readouterr()
    assert (
        jpp.main(
            [
                "visualize",
                "--model",
                str(tiny_model),
                "--lens",
                str(checkpoint),
                "--device",
                "cpu",
                "--local-files-only",
                "--prompt",
                "the cat",
                "--output",
                str(html),
            ]
        )
        == 0
    )
    assert "application/json" in html.read_text() and "<script src=" not in html.read_text()
    capsys.readouterr()
    assert jpp.main(["fit"]) == 2
    assert json.loads(capsys.readouterr().err.splitlines()[-1])["error"]["code"] == "usage_error"


def test_resident_jpp_tool_matches_reader(checkpoint, tiny_model):  # noqa: F811
    from jlens_cli.tools import LensTool

    tool = LensTool(
        {
            "backend": "jpp",
            "model": str(tiny_model),
            "lens": str(checkpoint),
            "device": "cpu",
            "dtype": "float32",
            "local_files_only": True,
            "max_tokens": 32,
        }
    )
    assert tool.handle({"id": 0, "method": "capabilities"})["data"]["backend"] == "jpp"
    reply = tool.handle(
        {
            "id": 1,
            "method": "readout",
            "params": {
                "samples": [{"text": "the cat sat ."}],
                "baseline": True,
                "top_k": 10,
                "track_token_ids": [3, 9],
            },
        }
    )
    assert reply["status"] == "ok", reply
    data = reply["data"]
    assert data["metrics"]["forward_calls"] == 1
    assert [row["kind"] for row in data["results"][0]["readouts"]] == ["jpp", "logit", "model"]
    cell = data["results"][0]["readouts"][0]["positions"][0]
    assert len(cell["top_tokens"]) == 9
    assert cell["tracked_tokens"][1]["filtered_rank"] is None
    args = jpp.parser().parse_args(
        [
            "apply",
            "--model",
            str(tiny_model),
            "--lens",
            str(checkpoint),
            "--prompt",
            "the cat sat .",
            "--top-k",
            "10",
        ]
    )
    jpp.resolve(args)
    expected = jpp.infer(
        args, tool.session.model, tool.session.lens, tool.session.metadata, args.prompt
    )
    for actual, reference in zip(cell["top_tokens"], expected["layers"]["0"][0], strict=True):
        assert actual["token_id"] == reference["token_id"]
        assert actual["probability"] == pytest.approx(reference["probability"])
