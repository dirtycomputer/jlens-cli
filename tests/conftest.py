from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    """Real Hugging Face decoder/tokenizer, random weights, no Hub download."""
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    torch.manual_seed(19)
    path = tmp_path_factory.mktemp("model")
    words = [
        "[UNK]",
        "[BOS]",
        "[EOS]",
        "the",
        "cat",
        "sat",
        "on",
        "a",
        "mat",
        "dog",
        "walked",
        "to",
        "park",
        "today",
        "we",
        "read",
        "books",
        "and",
        "write",
        "code",
        "is",
        "good",
        "hello",
        "world",
    ]
    backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", bos_token="[BOS]", eos_token="[EOS]"
    )
    tokenizer.save_pretrained(path)
    config = GPT2Config(
        vocab_size=len(words),
        n_positions=64,
        n_embd=8,
        n_layer=3,
        n_head=2,
        bos_token_id=1,
        eos_token_id=2,
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
    )
    model = GPT2LMHeadModel(config)
    model.save_pretrained(path)
    return path


@pytest.fixture
def run_cli():
    def run(*args, expect=0, stdin=None):
        env = dict(
            os.environ,
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false",
        )
        root = Path(__file__).resolve().parents[1]
        env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
        completed = subprocess.run(
            [sys.executable, "-m", "jlens_cli", *map(str, args)],
            capture_output=True,
            text=True,
            input=stdin,
            env=env,
            timeout=90,
        )
        assert completed.returncode == expect, completed.stderr
        return completed

    return run


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / "corpus.jsonl"
    prompts = ["the cat sat on a mat today", "the dog walked to the park today"]
    path.write_text(
        "".join(json.dumps({"id": i, "text": value}) + "\n" for i, value in enumerate(prompts))
    )
    return path
