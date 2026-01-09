#!/usr/bin/env python3
"""
SmolLM-135M decoder-only WebNN demo (greedy decode).

Loads a single WebNN graph from a local directory and runs a greedy decode loop
to generate text. Expected files:

    smollm-135.webnn
    smollm-135.weights
    smollm-135.manifest.json

By default, the script points to:
    /Users/tarekziade/Dev/SmolLM-135-webnn

Override with SMOLLM_WEBNN_DIR env var or --model-dir.

Notes:
- KV-cache is used if the graph exposes past_key_values inputs and present_* outputs.
- Input plumbing (attention_mask/position_ids) is detected from graph input names.
- This assumes an export similar to common causal-LM ONNX/WebNN patterns:
  - inputs: input_ids, attention_mask (optional), position_ids (optional), past_key_values_*
  - outputs: logits, present_*
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import webnn


HF_MODEL_ID = "HuggingFaceTB/SmolLM-135M"

DEFAULT_MODEL_DIR = Path(
    os.environ.get("SMOLLM_WEBNN_DIR", "/Users/tarekziade/Dev/SmolLM-135-webnn")
)

# Fallbacks if config doesn’t provide what we need (should, but keep robust).
FALLBACK_MAX_SEQ_LEN = 256
FALLBACK_DECODER_SEQ_LEN = 1


def _load_graph(
    graph_path: Path, weights_path: Path, manifest_path: Path
) -> webnn.MLGraph:
    """Load a WebNN graph from disk."""
    for path in (graph_path, weights_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    return webnn.MLGraph.load(
        str(graph_path),
        manifest_path=str(manifest_path),
        weights_path=str(weights_path),
    )


def _create_context(device: str) -> webnn.MLContext:
    """Create a WebNN context with a simple CPU/GPU toggle."""
    ml = webnn.ML()
    if device == "cpu":
        return ml.create_context(power_preference="default", accelerated=False)
    return ml.create_context(power_preference="high-performance", accelerated=True)


def _select_output(outputs: Dict[str, np.ndarray], desired: str) -> np.ndarray:
    """Return the named output, falling back to the first tensor if missing."""
    if desired in outputs:
        return outputs[desired]
    if outputs:
        return outputs[next(iter(outputs))]
    raise RuntimeError("No outputs returned from graph execution")


def _infer_kv_layout_from_config(cfg: AutoConfig) -> tuple[int, int, int]:
    """
    Return (num_layers, num_heads, head_dim) from HF config.
    For most decoder-only models: head_dim = hidden_size / num_attention_heads.
    """
    num_layers = int(getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 0)) or 0)
    num_heads = int(getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", 0)) or 0)
    hidden = int(getattr(cfg, "hidden_size", getattr(cfg, "n_embd", 0)) or 0)

    if not (num_layers and num_heads and hidden):
        raise ValueError(
            "Could not infer (num_layers, num_heads, hidden_size) from config. "
            "Please hardcode for your export."
        )
    head_dim = hidden // num_heads
    return num_layers, num_heads, head_dim


def _infer_max_seq_len(cfg: AutoConfig) -> int:
    # Different models use different names; keep it broad.
    for attr in ("max_position_embeddings", "n_positions", "seq_length"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    return FALLBACK_MAX_SEQ_LEN


def _make_inputs_for_step(
    graph_inputs: list[str],
    input_ids_step: np.ndarray,
    attention_mask_full: np.ndarray | None,
    position_ids_step: np.ndarray | None,
    past_inputs: Dict[str, np.ndarray] | None,
) -> Dict[str, np.ndarray]:
    """Build a feed dict based on the graph’s declared input names."""
    feed: Dict[str, np.ndarray] = {}

    # Common required input.
    if "input_ids" in graph_inputs:
        feed["input_ids"] = input_ids_step.astype(np.int64, copy=False)
    else:
        # Some exports might name it differently; fall back to first input.
        feed[graph_inputs[0]] = input_ids_step.astype(np.int64, copy=False)

    if attention_mask_full is not None and "attention_mask" in graph_inputs:
        feed["attention_mask"] = attention_mask_full.astype(np.int64, copy=False)

    if position_ids_step is not None and "position_ids" in graph_inputs:
        feed["position_ids"] = position_ids_step.astype(np.int64, copy=False)

    # Some exports use cache_position instead of position_ids.
    if position_ids_step is not None and "cache_position" in graph_inputs:
        # cache_position is often int64 scalar or [1]; we’ll provide [1]
        feed["cache_position"] = position_ids_step.reshape(-1).astype(
            np.int64, copy=False
        )

    # Past KV
    if past_inputs:
        for name, value in past_inputs.items():
            if name in graph_inputs:
                feed[name] = value

    return feed


def run_decoder_step(
    context: webnn.MLContext,
    graph: webnn.MLGraph,
    graph_inputs: list[str],
    input_ids_step: np.ndarray,
    attention_mask_full: np.ndarray | None,
    position_ids_step: np.ndarray | None,
    past_inputs: Dict[str, np.ndarray] | None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Execute the decoder graph and return logits and present key/values (if any)."""
    feed = _make_inputs_for_step(
        graph_inputs=graph_inputs,
        input_ids_step=input_ids_step,
        attention_mask_full=attention_mask_full,
        position_ids_step=position_ids_step,
        past_inputs=past_inputs,
    )

    result = context.compute(graph, feed)
    logits = _select_output(result, "logits")

    presents = {
        name: tensor
        for name, tensor in result.items()
        if name.startswith("present_")
        or name.startswith("past_key_values")  # be permissive
    }
    return logits, presents


def _detect_cache(
    graph_inputs: list[str], graph_outputs: list[str] | None = None
) -> bool:
    # Inputs typically include past_key_values_*
    if any("past_key_values" in n for n in graph_inputs):
        return True
    # Some exports use "past_key_" / "past_value_" naming.
    if any(n.startswith("past_") for n in graph_inputs):
        return True
    # If outputs are known, present_* indicates caching.
    if graph_outputs and any(n.startswith("present_") for n in graph_outputs):
        return True
    return False


def _init_past_kv(
    graph_inputs: list[str],
    num_layers: int,
    num_heads: int,
    head_dim: int,
    max_seq_len: int,
) -> Dict[str, np.ndarray]:
    """
    Initialize past KV tensors based on common naming patterns.

    We only allocate tensors for names that the graph actually expects.
    Shapes used here match a very common layout:
        (batch=1, num_heads, seq_len, head_dim) float32
    """
    past: Dict[str, np.ndarray] = {}

    # Pattern A (like many ONNX exports):
    #   past_key_values_{layer}_key
    #   past_key_values_{layer}_value
    for layer in range(num_layers):
        k_name = f"past_key_values_{layer}_key"
        v_name = f"past_key_values_{layer}_value"
        if k_name in graph_inputs:
            past[k_name] = np.zeros(
                (1, num_heads, max_seq_len, head_dim), dtype=np.float32
            )
        if v_name in graph_inputs:
            past[v_name] = np.zeros(
                (1, num_heads, max_seq_len, head_dim), dtype=np.float32
            )

    # Pattern B (some exports include "decoder" in name even for causal LM):
    for layer in range(num_layers):
        k_name = f"past_key_values_{layer}_decoder_key"
        v_name = f"past_key_values_{layer}_decoder_value"
        if k_name in graph_inputs:
            past[k_name] = np.zeros(
                (1, num_heads, max_seq_len, head_dim), dtype=np.float32
            )
        if v_name in graph_inputs:
            past[v_name] = np.zeros(
                (1, num_heads, max_seq_len, head_dim), dtype=np.float32
            )

    # Pattern C (past_{layer}_key / past_{layer}_value)
    for layer in range(num_layers):
        k_name = f"past_{layer}_key"
        v_name = f"past_{layer}_value"
        if k_name in graph_inputs:
            past[k_name] = np.zeros(
                (1, num_heads, max_seq_len, head_dim), dtype=np.float32
            )
        if v_name in graph_inputs:
            past[v_name] = np.zeros(
                (1, num_heads, max_seq_len, head_dim), dtype=np.float32
            )

    return past


def _update_past_from_presents(
    past_inputs: Dict[str, np.ndarray],
    presents: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Copy present_* tensors back into their matching past_* names when possible.

    We try a few mappings:
      present_{layer}_key -> past_key_values_{layer}_key
      present_{layer}_value -> past_key_values_{layer}_value
    and similar variants.
    """
    if not past_inputs:
        return past_inputs

    new_past = dict(past_inputs)

    for present_name, tensor in presents.items():
        # If the exporter already returns updated past_* tensors, accept them directly.
        if present_name in new_past:
            new_past[present_name] = tensor
            continue

        # Map present_* -> past_* by simple substitutions.
        mapped = present_name
        if mapped.startswith("present_"):
            mapped = mapped.replace("present_", "past_key_values_", 1)

        # common variants
        candidates = [
            mapped,
            mapped.replace("_decoder_", "_", 1),
            mapped.replace("past_key_values_", "past_", 1),
        ]

        for cand in candidates:
            if cand in new_past:
                new_past[cand] = tensor
                break

    return new_past


def greedy_decode(
    context: webnn.MLContext,
    graph: webnn.MLGraph,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    max_seq_len: int,
    debug: bool = False,
) -> str:
    graph_inputs = graph.get_input_names()
    # Some WebNN bindings also expose output names; if not, keep None.
    graph_outputs = getattr(graph, "get_output_names", lambda: None)()
    has_cache = _detect_cache(graph_inputs, graph_outputs)

    # Tokenize prompt without padding; we’ll manage our own fixed-size arrays if needed.
    enc = tokenizer(prompt, return_tensors="np", add_special_tokens=True)
    prompt_ids = enc["input_ids"].astype(np.int64)
    # prompt_ids shape: (1, prompt_len)
    prompt_len = int(prompt_ids.shape[1])

    eos_id = tokenizer.eos_token_id
    # Some tokenizers (GPT-ish) may not have pad; use eos as pad for fixed arrays.
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (eos_id or 0)
    )

    # Build attention mask matching the graph's expected shape
    # For models converted with past_sequence_length=0, attention_mask is [1, 1]
    # representing "attend to the current token"
    needs_attention_mask = "attention_mask" in graph_inputs
    attention_mask = None
    if needs_attention_mask:
        # Use [1, 1] for models with past_sequence_length=0
        # This represents attending to just the current token
        print("[INFO] Using attention_mask shape: [1, 1]")
        attention_mask = np.ones((1, 1), dtype=np.int64)

    needs_position_ids = ("position_ids" in graph_inputs) or (
        "cache_position" in graph_inputs
    )

    # For first step, many exports accept full prompt; others want only last token + cache.
    # We’ll do:
    # - Step 0: feed the full prompt (clipped/padded) if not using cache,
    #           else we "prefill" by iterating through prompt tokens.
    cfg = AutoConfig.from_pretrained(HF_MODEL_ID)
    num_layers, num_heads, head_dim = _infer_kv_layout_from_config(cfg)

    past_inputs: Dict[str, np.ndarray] = {}
    if has_cache:
        past_inputs = _init_past_kv(
            graph_inputs=graph_inputs,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
        )

    generated: list[int] = [int(x) for x in prompt_ids[0].tolist()]

    def step(token_id: int, pos: int) -> int:
        nonlocal past_inputs, attention_mask

        # seq_len=1 token input for incremental decode
        input_step = np.array([[token_id]], dtype=np.int64)

        position_ids_step = None
        if needs_position_ids:
            # Most models use 0-based positions
            position_ids_step = np.array([[pos]], dtype=np.int64)

        # For models with fixed attention_mask shape [1,1], no need to update
        # For models with dynamic masks, update the position
        if attention_mask is not None and attention_mask.shape[1] > 1:
            if pos < attention_mask.shape[1]:
                attention_mask[0, pos] = 1

        logits, presents = run_decoder_step(
            context=context,
            graph=graph,
            graph_inputs=graph_inputs,
            input_ids_step=input_step,
            attention_mask_full=attention_mask,
            position_ids_step=position_ids_step,
            past_inputs=past_inputs if has_cache else None,
        )

        # logits commonly shaped [1, 1, vocab] or [1, seq, vocab]
        if logits.ndim == 3:
            last_logits = logits[0, -1]
        elif logits.ndim == 2:
            last_logits = logits[0]
        else:
            raise RuntimeError(f"Unexpected logits shape: {logits.shape}")

        if debug:
            topk = np.argsort(last_logits)[-5:][::-1]
            decoded_topk = [tokenizer.decode([int(t)]) for t in topk]
            print(
                f"[DEBUG] pos={pos} top5: "
                + ", ".join(f"{int(t)}='{s}'" for t, s in zip(topk, decoded_topk))
            )

        next_token = int(np.argmax(last_logits))

        if has_cache and presents:
            past_inputs = _update_past_from_presents(past_inputs, presents)

        return next_token

    # Prefill: if cache, run through the prompt tokens (except maybe last) to build KV.
    # This is slower but robust across export styles.
    if has_cache:
        # Reset mask to cover prompt (only for dynamic masks)
        if attention_mask is not None and attention_mask.shape[1] > 1:
            attention_mask[:] = 0
            attention_mask[0, : min(prompt_len, attention_mask.shape[1])] = 1

        # Push tokens one-by-one; last token becomes the context for generation.
        for pos in range(min(prompt_len, max_seq_len)):
            _ = step(token_id=generated[pos], pos=pos)

    # Generate new tokens.
    start_pos = min(prompt_len, max_seq_len)
    for i in range(max_new_tokens):
        pos = min(start_pos + i, max_seq_len - 1)
        next_tok = step(token_id=generated[-1], pos=pos)
        generated.append(next_tok)
        if eos_id is not None and next_tok == eos_id:
            break

    return tokenizer.decode(generated, skip_special_tokens=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SmolLM-135M WebNN decoder-only demo")
    p.add_argument(
        "--prompt",
        default="Write a short haiku about on-device AI.",
        help="Prompt text",
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory containing smollm-135.{webnn,weights,manifest.json}",
    )
    p.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Preferred device hint for MLContext",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=48,
        help="Maximum tokens to generate",
    )
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=0,
        help="Override max sequence length (0 = infer from config)",
    )
    p.add_argument(
        "--skip-reference",
        action="store_true",
        help="Skip running the transformers reference model",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose decoding debug output (top-5 tokens per step)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    cfg = AutoConfig.from_pretrained(HF_MODEL_ID)
    max_seq_len = args.max_seq_len or _infer_max_seq_len(cfg)

    # Transformers reference for sanity (CPU).
    if not args.skip_reference:
        print("[INFO] Running transformers reference (PyTorch, CPU)...")
        ref_model = AutoModelForCausalLM.from_pretrained(HF_MODEL_ID)
        ref_inp = tokenizer(args.prompt, return_tensors="pt")
        ref_out = ref_model.generate(
            **ref_inp,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
        ref_text = tokenizer.decode(ref_out[0], skip_special_tokens=True)
        print("[OK] Transformers output:")
        print(ref_text.strip())
    else:
        ref_text = None

    graph = _load_graph(
        args.model_dir / "smollm-135.webnn",
        args.model_dir / "smollm-135.weights",
        args.model_dir / "smollm-135.manifest.json",
    )
    graph_inputs = graph.get_input_names()
    print(f"[INFO] Graph inputs: {graph_inputs}")
    if hasattr(graph, "get_output_names"):
        print(f"[INFO] Graph outputs: {graph.get_output_names()}")

    context = _create_context(args.device)
    print(f"[INFO] WebNN context created (accelerated={context.accelerated})")
    print(f"[INFO] Using max_seq_len={max_seq_len}")

    out_text = greedy_decode(
        context=context,
        graph=graph,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        max_seq_len=max_seq_len,
        debug=args.debug,
    )

    print("\n=== SmolLM-135M WebNN output ===")
    print(out_text.strip())

    if ref_text is not None:
        print("\n=== Reference vs WebNN ===")
        print(f"Reference: {ref_text.strip()}")
        print(f"WebNN    : {out_text.strip()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
