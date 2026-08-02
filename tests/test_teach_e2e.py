#!/usr/bin/env python3
"""End-to-end prompt-learning test — GPU required (manual §7 recipe).

Teaches a few facts, reloads the store fresh (proving persistence without the
original prompt in any window), and checks role-intersection retrieval.

Run:
    python3 tests/test_teach_e2e.py
"""

import gc
import os
import shutil
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from figtree import connect, retrieve_by_roles, teach

MODEL_ID = "unsloth/Qwen3-4B-bnb-4bit"
STORE_URI = "/tmp/figtree_learn_e2e.lance"

TEACHINGS = [
    "My daughter's name is June and she is allergic to cashews.",
    "Never use the legacy auth endpoint again.",
    "My name is Alex. Please remember that.",
]


def main() -> bool:
    if Path(STORE_URI).exists():
        shutil.rmtree(STORE_URI)

    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    store = connect(STORE_URI)

    # Session 1: teach the facts.
    print("\nTeaching facts...")
    for i, stmt in enumerate(TEACHINGS, 1):
        res = teach(
            model, tokenizer, stmt, store, session_id="user", trust=0.95,
        )
        print(f"  {i}. roles={res['roles']} created={res['created']} superseded={res['superseded']}")
        if not res["roles"]:
            print(f"     WARNING: no roles extracted (decode={res['decode_output']!r})")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Session 2: fresh load — the original prompts are NOT in any window.
    print("\nReloading store fresh (prompts not in window)...")
    store = connect(STORE_URI)
    assert store.count() > 0, "store empty after reload"

    ok = True
    for query in [
        "What is the user's name?",
        "What should I do instead of the legacy auth endpoint?",
        "What is the user's daughter allergic to?",
    ]:
        hits = retrieve_by_roles(query, store)  # rule-based, no model needed
        print(f"\nQuery: {query}")
        print(f"  retrieved {len(hits)} statement(s):")
        for f in hits:
            print(f"    [{f.meta.get('role_match')}] {f.text}")
        if not hits:
            ok = False
            print("  FAILED: no learned statement retrieved")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model.cpu()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nPASS" if ok else "\nFAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
