from __future__ import annotations

import copy
import io
import json
import os
import subprocess
import sys

import pytest

from jlens_cli import analysis
from jlens_cli.io import CLIError, digest
from jlens_cli.tools import LensTool, serve


@pytest.fixture
def tool_config(tiny_model, tmp_path):
    import torch
    from jlens import JacobianLens

    lens = tmp_path / "lens.pt"
    JacobianLens({0: torch.eye(8), 1: torch.eye(8) * 0.5}, n_prompts=5, d_model=8).save(
        str(lens), dtype=torch.float32
    )
    return {
        "model": str(tiny_model),
        "lens": str(lens),
        "device": "cpu",
        "dtype": "float32",
        "local_files_only": True,
        "max_tokens": 32,
        "max_samples": 8,
        "max_readout_cells": 64,
    }


def call(tool, method, params=None, *, request_id=1, **extra):
    reply = tool.handle({"id": request_id, "method": method, "params": params or {}, **extra})
    assert reply["status"] == "ok", reply.get("error", reply)
    return reply["data"]


def rehash(data):
    data["manifest_id"] = digest(data["manifest"])
    data["artifact_id"] = digest(
        {k: v for k, v in data.items() if k not in ("artifact_id", "metrics")}
    )
    return data


def observation(tool):
    return call(
        tool,
        "readout",
        {
            "samples": [{"id": "a", "text": "the cat sat"}],
            "top_k": 3,
            "concepts": {"need": ["cat"], "wrong": ["dog"]},
            "baseline": True,
        },
    )


def test_resident_readout_matches_upstream_and_reuses_forward(tool_config):
    import torch

    tool = LensTool(tool_config)
    assert not call(tool, "capabilities")["loaded"]
    first = tool.handle(
        {
            "id": 1,
            "method": "readout",
            "params": {
                "samples": [{"text": "the cat sat"}],
                "baseline": True,
                "track_token_ids": [4],
                "top_k": 5,
            },
        }
    )
    assert first["status"] == "ok" and first["metrics"]["model_loaded_this_request"]
    data = first["data"]
    assert data["metrics"]["forward_calls"] == 1
    sample = data["results"][0]
    direct, final, _ = tool.session.lens.apply(
        tool.session.model, "the cat sat", positions=sample["positions"]
    )
    for readout in sample["readouts"]:
        if readout["kind"] == "logit":
            continue
        logits = final if readout["kind"] == "model" else direct[readout["layer"]]
        values, ids = logits.softmax(-1).topk(5)
        top = readout["positions"][0]["top_tokens"]
        assert [v["token_id"] for v in top] == ids[0].tolist()
        torch.testing.assert_close(torch.tensor([v["probability"] for v in top]), values[0])
    second = tool.handle(
        {
            "id": 2,
            "method": "readout",
            "params": {
                "samples": [{"input_ids": sample["input_ids"]}],
                "baseline": True,
                "track_token_ids": [4],
                "top_k": 5,
            },
        }
    )
    assert second["status"] == "ok" and not second["metrics"]["model_loaded_this_request"]
    assert second["data"]["results"][0]["readouts"] == sample["readouts"]
    assert second["data"]["manifest_id"] == data["manifest_id"]
    stale = tool.handle(
        {
            "id": 3,
            "method": "input.prepare",
            "params": {"sample": {"text": "cat"}},
            "manifest_id": "wrong",
        }
    )
    assert stale["error"]["code"] == "stale_manifest"


def test_inputs_concepts_truncation_and_budgets(tool_config):
    tool = LensTool(tool_config)
    rejected = tool.handle(
        {"id": 0, "method": "readout", "params": {"samples": [{"text": "cat"}] * 9}}
    )
    assert rejected["error"]["code"] == "budget_exceeded"
    assert tool.session is None and not rejected["metrics"]["model_loaded_this_request"]
    long = {"text": " ".join(["the"] * 20 + ["cat"]), "max_seq_len": 8}
    reply = tool.handle({"id": 1, "method": "input.prepare", "params": {"sample": long}})
    assert reply["error"]["code"] == "input_truncated"
    prepared = call(tool, "input.prepare", {"sample": {**long, "truncation": "left"}})
    count = len(tool.session.model.tokenizer(long["text"], truncation=False)["input_ids"])
    assert prepared["truncated"] and prepared["original_num_tokens"] == count
    assert (
        prepared["original_positions"] == [count - 1] and prepared["tokens"][-1]["token"] == "cat"
    )
    assert prepared["predicts_original_positions"] == [count]
    concepts = call(
        tool,
        "concepts.resolve",
        {"concepts": {"ok": ["cat"], "mixed": ["cat", "cat dog"], "bad": ["cat dog"]}},
    )["concepts"]
    assert concepts["ok"]["coverage"] == "supported"
    assert concepts["mixed"]["coverage"] == "partial"
    assert concepts["bad"]["coverage"] == "unsupported"
    masked = call(
        tool,
        "readout",
        {"samples": [{"input_ids": [3, 4], "attention_mask": [0, 1], "position_ids": [0, 0]}]},
    )
    assert masked["results"][0]["input_ids"] == [3, 4]
    invalid = tool.handle(
        {
            "id": 2,
            "method": "readout",
            "params": {
                "samples": [{"input_ids": [3, 4], "attention_mask": [0, 1], "positions": [0]}]
            },
        }
    )
    assert invalid["status"] == "error"
    too_large = tool.handle(
        {"id": 3, "method": "readout", "params": {"samples": [{"text": "cat"}] * 9}}
    )
    assert too_large["error"]["code"] == "budget_exceeded"
    logit = call(tool, "readout", {"samples": [{"text": "cat"}], "kind": "logit", "layers": [2]})
    assert logit["results"][0]["readouts"][0]["layer"] == 2


def test_component_projection_is_manual_contrast_and_does_not_edit(tool_config):
    import torch

    tool = LensTool(tool_config)
    modules = call(tool, "components.list")["modules"]
    name = next(m["name"] for m in modules if m["kind"] == "mlp")
    model, lens = tool.session.model, tool.session.lens
    before = {n: p.clone() for n, p in model._hf_model.named_parameters()}
    captured = []
    module = dict(model._hf_model.named_modules())[name]
    handle = module.register_forward_hook(
        lambda _m, _a, output: captured.append(output[0, -1].detach())
    )
    try:
        for text in ("the cat sat", "the dog walked"):
            model.forward(model.encode(text))
    finally:
        handle.remove()
    direction = (model._lm_head.weight[4] - model._lm_head.weight[9]).float() @ lens.jacobians[0]
    direction /= direction.norm()
    expected = ((captured[0] - captured[1]) @ direction).item()
    report = call(
        tool,
        "components.score",
        {
            "modules": [name],
            "failure_samples": [{"text": "the cat sat"}],
            "reference_samples": [{"text": "the dog walked"}],
            "concepts": {"need": [4], "wrong": [9]},
        },
    )
    assert report["modules"][0]["score"] == pytest.approx(expected, abs=1e-7)
    assert report["metrics"]["forward_calls"] == 2
    assert report["evidence_type"] == "associational_linear_projection"
    assert not module._forward_hooks
    for n, p in model._hf_model.named_parameters():
        assert torch.equal(p, before[n]) and not p.requires_grad


def test_artifact_analysis_checks_comparability_and_unknowns(tool_config):
    tool = LensTool(tool_config)
    data = observation(tool)
    result = call(tool, "concepts.trajectory", {"observation": data})
    assert len(result["trajectories"]) == 2
    identical = call(tool, "checkpoint.compare", {"before": data, "after": data})
    assert identical["mean_top_k_overlap"] == 1
    assert all(change["rank_delta"] == 0 for change in identical["tracked_changes"])
    audit = call(tool, "audit.scan", {"observation": data})
    assert audit["signals"][0]["supported_samples"] == 1
    assert audit["decision"] == "external_behavioral_gate_required"
    missing = copy.deepcopy(data)
    missing["concepts"]["need"]["unsupported"] = [{"alias": "cat dog"}]
    rehash(missing)
    assert analysis.trajectory({"observation": missing})["trajectories"][0]["pattern"] == "unknown"
    changed = copy.deepcopy(data)
    changed["results"][0]["input_hash"] = "other"
    rehash(changed)
    with pytest.raises(CLIError, match="Fixed replay"):
        analysis.compare({"before": data, "after": changed})
    changed["results"][0]["input_hash"] = "tampered-after-hash"
    with pytest.raises(CLIError, match="artifact ID"):
        analysis.artifact(changed)


def test_health_uses_only_transport_and_requires_independent_reference(tool_config):
    data = observation(LensTool(tool_config))
    params = {"reference": data, "candidate": data, "min_overlap": 0.8, "min_cells": 1}
    assert analysis.health(params)["status"] == "unknown"
    reference = copy.deepcopy(data)
    reference["manifest"]["lens_sha256"] = "fresh-reference"
    reference["manifest"]["lens_calibration_identity"] = reference["manifest"]["artifact_identity"]
    rehash(reference)
    assert analysis.health({**params, "reference": reference})["status"] == "consistent"
    candidate = copy.deepcopy(data)
    for row in candidate["results"][0]["readouts"]:
        if row["kind"] == "jacobian":
            for token in row["positions"][0]["top_tokens"]:
                token["token_id"] += 1000
    rehash(candidate)
    result = analysis.health({**params, "reference": reference, "candidate": candidate})
    assert result["status"] == "degraded" and result["mean_top_k_overlap"] == 0


def test_canary_health_does_not_require_refitting(tool_config):
    tool = LensTool(tool_config)
    data = observation(tool)
    params = {
        "observation": data,
        "canaries": [{"sample_id": "a", "expected_token_ids": [4]}],
        "rank_threshold": 25,
        "min_pass_rate": 1.0,
        "min_samples": 1,
    }
    result = call(tool, "lens.health", params)
    assert result["mode"] == "labeled_canary" and result["status"] == "consistent"
    assert result["pass_rate"] == 1
    unknown = call(
        tool,
        "lens.health",
        {**params, "canaries": [{"sample_id": "a", "expected_token_ids": [999]}]},
    )
    assert unknown["status"] == "unknown"
    assert unknown["supported_samples"] == 0


def test_bounded_trial_candidates_need_explicit_algorithm_and_whitelist():
    report = {
        "manifest_id": "checkpoint-x",
        "modules": [
            {
                "name": f"model.layers.{i}.mlp.down_proj",
                "layer": i,
                "kind": "mlp",
                "score": 4 - i,
                "lora_params_per_rank": 32,
            }
            for i in range(4)
        ],
    }
    params = {
        "component_report": report,
        "allowed_modules": [r["name"] for r in report["modules"]],
        "algorithm": "grpo",
        "training_data_id": "data-x",
        "rank": 2,
        "max_trainable_parameters": 128,
    }
    result = analysis.propose(params)
    assert 2 <= len(result["candidates"]) <= 3
    for plan in result["candidates"]:
        assert plan["estimated_trainable_parameters"] <= 128
        assert set(plan["peft_config"]["target_modules"]) <= set(params["allowed_modules"])
        assert plan["state"] == "unvalidated_trial_candidate"
    with pytest.raises(CLIError):
        analysis.propose({**params, "algorithm": "automatic"})
    crt = analysis.propose({**params, "algorithm": "crt"})
    assert all(c["supervision"]["scope"] == "reflection" for c in crt["candidates"])
    selected = analysis.propose(
        {**params, "loss_scope": "selected_positions", "loss_positions": {"case": [2, 3]}}
    )
    assert selected["candidates"][0]["supervision"]["loss_positions"] == {"case": [2, 3]}


def test_worker_protocol_errors_continue_and_close():
    requests = (
        "\n".join(
            [
                json.dumps({"id": 1, "method": "capabilities"}),
                "not json",
                json.dumps({"id": 2, "method": "execute_training"}),
                json.dumps({"id": 3, "method": "close"}),
            ]
        )
        + "\n"
    )
    output = io.StringIO()
    assert serve(LensTool(), io.StringIO(requests), output) == 0
    replies = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [r["status"] for r in replies] == ["ok", "error", "error", "ok"]
    assert replies[2]["error"]["code"] == "unsupported_method"
    output = io.StringIO()
    serve(
        LensTool(),
        io.StringIO("x" * 50 + "\n" + json.dumps({"id": 1, "method": "close"}) + "\n"),
        output,
        max_request_bytes=40,
    )
    assert json.loads(output.getvalue().splitlines()[0])["error"]["code"] == "budget_exceeded"
    output = io.StringIO()
    serve(LensTool(), io.StringIO('{"id":NaN,"method":"capabilities"}\n'), output)
    assert json.loads(output.getvalue())["error"]["code"] == "invalid_json"


def test_many_tracked_ranks_preserve_ties_and_filtering():
    import torch

    from jlens_cli.operations import rows

    class Tokens:
        def decode(self, ids):
            return str(ids[0])

    logits = torch.tensor([[0.0, 2.0, 1.0, 2.0, -1.0, 1.0, 0.0, 3.0, 2.0, 4.0]])
    mask = torch.tensor([True, False] * 5)
    cell = rows(logits, [0], [0], Tokens(), 3, list(range(10)), mask)[0]
    for item in cell["tracked_tokens"]:
        token = item["token_id"]
        assert item["rank"] == int((logits[0] > logits[0, token]).sum()) + 1
        expected = int(((logits[0] > logits[0, token]) & mask).sum()) + 1 if mask[token] else None
        assert item["filtered_rank"] == expected


def test_messages_are_explicitly_templated(tool_config, tmp_path):
    import shutil

    from transformers import AutoTokenizer

    model = tmp_path / "templated-model"
    shutil.copytree(tool_config["model"], model)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}"
    tokenizer.save_pretrained(model)
    tool = LensTool({**tool_config, "model": str(model)})
    value = call(
        tool,
        "input.prepare",
        {
            "sample": {
                "messages": [{"role": "user", "content": "the cat"}],
                "add_generation_prompt": False,
            }
        },
    )
    assert value["input_kind"] == "messages"
    assert value["input_ids"] == tokenizer.apply_chat_template(
        [{"role": "user", "content": "the cat"}],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=False,
    )


def test_one_shot_tool_cannot_overwrite_model_or_lens(tool_config, tmp_path, capsys):
    from jlens_cli.cli import main

    config = tmp_path / "config.json"
    config.write_text(json.dumps(tool_config))
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"id": 1, "method": "capabilities"}))
    lens = __import__("pathlib").Path(tool_config["lens"])
    before = lens.read_bytes()
    assert (
        main(
            [
                "tool",
                "--config",
                str(config),
                "--request",
                str(request),
                "--output",
                str(lens),
                "--force",
            ]
        )
        == 2
    )
    assert "overwrite input" in capsys.readouterr().err
    assert lens.read_bytes() == before


def test_real_worker_cli_loads_once(tool_config, tmp_path):
    config = tmp_path / "session.json"
    config.write_text(json.dumps(tool_config))
    requests = [
        {"id": i, "method": "readout", "params": {"samples": [{"text": "the cat"}]}} for i in (1, 2)
    ] + [{"id": 3, "method": "close"}]
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    result = subprocess.run(
        [sys.executable, "-m", "jlens_cli", "serve", "--config", str(config)],
        input="".join(json.dumps(r) + "\n" for r in requests),
        text=True,
        capture_output=True,
        env=env,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    replies = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(replies) == 3 and all(r["status"] == "ok" for r in replies)
    assert replies[0]["metrics"]["model_loaded_this_request"]
    assert not replies[1]["metrics"]["model_loaded_this_request"]
    assert replies[0]["data"]["artifact_id"] == replies[1]["data"]["artifact_id"]
