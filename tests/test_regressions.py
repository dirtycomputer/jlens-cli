from __future__ import annotations

import json
import shutil

import pytest

from jlens_cli.cli import main


@pytest.mark.parametrize("backend", ["jacobian", "jpp"])
@pytest.mark.parametrize("sidecar", [False, True])
def test_directory_alias_and_metadata_cannot_be_overwritten(tmp_path, capsys, backend, sidecar):
    import torch
    from jlens import JacobianLens

    lens = tmp_path / "lens.pt"
    if backend == "jacobian":
        JacobianLens({0: torch.eye(8)}, n_prompts=1, d_model=8).save(str(lens))
    else:
        torch.save(
            {
                "parameters": {"jacobians": {0: torch.eye(8)}},
                "source_layers": [0],
                "config": {
                    "hf_model_name": "local",
                    "d_model": 8,
                    "num_prompts_trained_on": 1,
                    "lens_type": "jacobian",
                    "relative_end_transport_layer": -1,
                },
            },
            lens,
        )
    target = tmp_path / "lens.pt.meta.json" if sidecar else lens
    if sidecar:
        target.write_text('{"model":"original"}')
    before = target.read_bytes()
    assert (
        main(
            ["inspect", "--lens", str(tmp_path), "--output", str(target), "--force"],
            backend=backend,
        )
        == 2
    )
    assert "overwrite input" in capsys.readouterr().err
    assert target.read_bytes() == before


def test_recursive_merge_checks_original_leaves(tmp_path, capsys):
    import torch
    from jlens import JacobianLens

    def make(name, value, skip=0):
        path = tmp_path / (name + ".pt")
        JacobianLens({0: torch.eye(8) * value}, n_prompts=1, d_model=8).save(
            str(path), dtype=torch.float32
        )
        metadata = {
            "model": "model",
            "revision": "commit",
            "fit": {
                "model": "model",
                "revision": "commit",
                "layers": [0],
                "skip_first": skip,
                "prompts_sha256": name,
                "num_shards": 1,
                "shard_index": 0,
            },
        }
        path.with_name(path.name + ".meta.json").write_text(json.dumps(metadata))
        return str(path)

    a, b, c = make("a", 1), make("b", 2), make("c", 3, skip=1)
    ab = str(tmp_path / "ab.pt")
    assert main(["merge", a, b, "-o", ab]) == 0
    capsys.readouterr()
    assert main(["merge", ab, a, "-o", str(tmp_path / "aba.pt")]) == 2
    assert "repeated prompt subsets" in capsys.readouterr().err
    assert main(["merge", ab, c, "-o", str(tmp_path / "abc.pt")]) == 2
    assert "different estimator" in capsys.readouterr().err
    d = make("d", 4)
    __import__("pathlib").Path(d + ".meta.json").write_text('{"model":"model","revision":"commit"}')
    assert main(["merge", a, d, "-o", str(tmp_path / "incomplete.pt")]) == 0
    assert not json.loads(capsys.readouterr().out)["data"]["provenance_complete"]


def test_local_content_changes_block_resume(tiny_model, tmp_path, capsys):
    import torch
    from safetensors.torch import load_file, save_file

    model = tmp_path / "model"
    shutil.copytree(tiny_model, model)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("the cat sat on a mat today\n")
    args = [
        "fit",
        "--model",
        str(model),
        "--prompts",
        str(corpus),
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--local-files-only",
        "--skip-first",
        "0",
        "--max-seq-len",
        "32",
        "--dim-batch",
        "4",
        "--checkpoint",
        str(tmp_path / "checkpoint.pt"),
        "--output",
        str(tmp_path / "lens.pt"),
    ]
    assert main(args) == 0
    capsys.readouterr()
    weights = model / "model.safetensors"
    tensors = load_file(str(weights))
    tensors[next(iter(tensors))] = tensors[next(iter(tensors))] + torch.ones_like(
        tensors[next(iter(tensors))]
    )
    save_file(tensors, str(weights), metadata={"format": "pt"})
    assert main([*args, "--force"]) == 2
    assert "checkpoint_mismatch" in capsys.readouterr().err


def test_cli_logit_only_can_read_unfitted_layer(tiny_model, tmp_path, capsys):
    import torch
    from jlens import JacobianLens

    lens = tmp_path / "lens.pt"
    JacobianLens({0: torch.eye(8)}, n_prompts=1, d_model=8).save(str(lens))
    args = [
        "apply",
        "--model",
        str(tiny_model),
        "--lens",
        str(lens),
        "--device",
        "cpu",
        "--local-files-only",
        "--prompt",
        "the cat",
        "--layers",
        "2",
        "--logit-lens",
    ]
    assert main(args) == 0
    assert (
        json.loads(capsys.readouterr().out)["data"]["results"][0]["readouts"][0]["kind"] == "logit"
    )
    assert main([*args, "--prompt", " ".join(["the"] * 40), "--max-seq-len", "8"]) == 2
    assert "input_truncated" in capsys.readouterr().err
