"""Numerical checkpoint compatibility and CLI boundary checks, entirely offline."""

import json

import pytest
from test_integration import tiny_model  # noqa: F401

from jlens_cli import cli, jpp

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
    assert jpp.main(["inspect", "--lens", str(checkpoint)]) == 0
    assert json.loads(capsys.readouterr().out)["d_model"] == 8
    assert (
        jpp.main(
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
        jpp.main(["inspect", "--lens", str(checkpoint), "--output", str(checkpoint), "--force"])
        == 1
    )
