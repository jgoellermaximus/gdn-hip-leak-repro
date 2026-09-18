# Hybrid Gated-DeltaNet models on ROCm/HIP: fused GDN op carries recurrent state across requests on a reused `llama-server` slot

Earlier prompts' text leaks into later completions. The leak survives both `memory_seq_rm [0, end)` (full re-process from position 0) and a
context-checkpoint restore, and disappears only when the fused Gated-DeltaNet op is disabled. This repository is the minimal, self-contained
reproduction: one bare `llama-server`, one slot, three `/completion` requests over synthetic documents, plus the logs it produced on our box.

## Affected versions (as tested)

| bundle | llama.cpp build | ggml | status |
|---|---|---|---|
| **ollama 0.32.14** (`llama-server`) | `7e4c0a968` (`common_params_print_info: build 1 (7e4c0a968) with GNU 13.3.1`) | `libggml.so.0.20.0` | **reproduced** — the 3-request repro below, both memory paths, two models |
| **ollama 0.34.1** (`llama-server.real`, `(version 0.34.1)`) | `5d806aa25` | `libggml.so.0.23.0`, `libllama.so.0.4.0` | **defect still present** on a longer sequence of unrelated documents (8 cross-document leaks + 1 runaway with all layers on GPU, request-exact identical on two runs; a residual 1-in-29 leak even with layer 0 on CPU). This version's resolver logs no `fused Gated Delta Net` lines for these models, so we make no kernel attribution for it — "defect present" only. The document set for that test is private; only this summary is included |

Both bundles ship the same `gated_delta_net_cuda<…>` kernel set in `rocm_v7_2/libggml-hip.so`. No env/flag disables the fused GDN op;
`GGML_CUDA_DISABLE_FUSION` does not reach it on 0.32.14.

## Environment

- GPU: AMD Radeon 8060S (Strix Halo, **gfx1151**), integrated; ROCm backend `rocm_v7_2` (`libamdhip64.so.7.2.70201`),
  `GGML_BACKEND_PATH=…/rocm_v7_2/libggml-hip.so`
- CPU: AMD RYZEN AI MAX+ 395; 122 GiB RAM; `amdgpu.gttsize=126976` (ROCm0 reports 126976 MiB, ~101 GiB free at load)
- Linux `7.0.0-31-generic`
- Models (Q4_K_M GGUF as pulled by ollama): `qwen3.6:35b-a3b` (arch `qwen35moe`, hybrid GDN + attention, blob
  `sha256-f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d`) and `qwen3.8:27b` (arch `qwen35`, dense hybrid GDN + attention,
  blob `sha256-f5f1dd8920d417aac2718b0bda3403da274301efdd6760b4f0f4b864ff2ad57d`)
- Server flags (ollama's production runner line, reproduced verbatim on the bare server): `-c 24576 -np 1 -b 1024 -ub 1024 --cache-type-k q8_0
  --cache-type-v q8_0 --flash-attn on --context-shift --keep 4 --load-mode dio`

## Minimal repro

Scripts: `rc3/leak_probe2.py` (driver + proof capture) and `rc3/synth_docs.py` (three deterministic synthetic documents with **disjoint
vocabularies** — "Harbor Lantern Charter" / "Velvet Gantry Ledger Rules" / "Ember Rampart Review Procedure" — so any leak is attributable by
inspection). No real documents are involved.

```
export LD_LIBRARY_PATH=<ollama lib>:<ollama lib>/rocm_v7_2 GGML_BACKEND_PATH=<ollama lib>/rocm_v7_2/libggml-hip.so
python rc3/leak_probe2.py --blob <qwen3.6-35b-a3b.gguf> --ngl 42 --out fused_on     # leaks
python rc3/leak_probe2.py --blob <qwen3.6-35b-a3b.gguf> --ngl 41 --out fused_off    # does not leak
```

The driver starts `llama-server --model <gguf> --port 18080 --host 127.0.0.1 --no-webui --offline -c 24576 -np 1 --log-verbosity 4 --load-mode dio
--cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on -b 1024 -ub 1024 --context-shift --keep 4 -ngl N`, waits for `/health`, then posts three
`/completion` requests (`temperature 0`, `seed 7`, `cache_prompt true`, `n_predict 300`), ChatML with a think-off prefill
(`<|im_start|>assistant\n<think>\n\n</think>\n\n`), and a generic system prompt S ("extractive one-line abstract + one verbatim quote, JSON only"):

| request | prompt | tokens | server path (from the verbose log) | expected | observed, fused ON (`-ngl 42`) |
|---|---|---|---|---|---|
| R1 | S + document **A** | 1,055 | fresh slot, `memory_seq_rm [0, end)`; a context checkpoint lands at pos 30 (inside S) | abstract + quote of A | abstract + quote of A (correct) |
| R2 | S + document **B1** (same S prefix) | 4,270 | **`restored context checkpoint (pos_min = 30, pos_max = 30, n_tokens = 31, n_past = 31)`** → B1 processed from pos 31 | abstract + quote of B1 | **abstract of A + a verbatim 40-char clause of A** |
| R3 | `<nonce> ` + S + document **B2** (prefix broken on purpose) | 3,268 | checkpoint checks fail (`checking checkpoint with [30, 30] against 3…`) → **`cached n_tokens = 0, memory_seq_rm [0, end)`**, full re-process from position 0 | abstract + quote of B2 | **still A's abstract + a verbatim A clause** |

So the leak survives **both** memory paths a reused slot can take — the checkpoint-restore path (R2) and the n_past=0 / `seq_rm [0, end)` full
re-process path (R3). Deterministic (identical outputs across passes at temperature 0). Nothing in the prompt of R2/R3 contains A's vocabulary.

**Control — same binary, same flags, `-ngl 41` (layer 0 on CPU):** the resolver prints `fused Gated Delta Net (autoregressive) not supported, set to
disabled; fused Gated Delta Net (chunked) not supported, set to disabled` and the leak is gone (R3 → correct B2 abstract; R1/R2 on this run returned an
empty completion — a separate think-off/EOS-first behaviour of the unfused graph, noted under "additional observations"). Layer-0-on-CPU is the
*only* available switch for the op. Placement itself is not the variable: moving 24 layers' experts to host memory with the op still enabled leaves
the leak signature identical, while moving only layer 0 (793 MiB) with the op disabled is clean.

**Same result on the dense sibling:** `qwen3.8:27b` `-ngl 66` (66/66, both paths enabled) leaks on R2 and R3 (verbatim A clauses in both); `-ngl 65`
(65/66, both paths disabled) does not.

## Observed vs expected

- **Expected:** each `/completion` on a slot is independent once the slot's memory is cleared (`memory_seq_rm [0, end)`) or restored to a checkpoint that
  predates the previous document; a hybrid model's recurrent (GDN) state should be re-initialised from the zeroed state (`state_zero` / `get_rows` of a
  fresh cell) on every such request.
- **Observed:** with the fused GDN op enabled on HIP, the recurrent state of the previous request persists into the next one — the first ubatch of R2/R3
  reads state written by R1, and the model emits R1's document text as "verbatim evidence" for R2/R3. On longer sequences (29 unrelated documents on one
  warm runner, `-ub 1024`) the carried state accumulates until generation degrades into unterminated JSON ("runaway") — ubatch size moves the onset
  (`-ub 512`: ≈15 requests; `-ub 1024`: 2–3), not the mechanism. Graph reuse, HIP graphs, `GGML_CUDA_DISABLE_FUSION`, expert placement,
  `--ctx-checkpoints 0`, `--cache-ram 0`, `--context-shift`, f16 KV cache and `OLLAMA_NUM_PARALLEL` (refused for these archs) were each varied
  one at a time over 22 single-variable server configurations: the leak persists in all of them; only disabling the fused op clears it.

Where the state physically persists inside the fused op (a persistent workspace of `ggml-cuda/gated_delta_net.cu`, or the fused kernel reading the RS
cell directly and bypassing the zeroed input) we could not pin from outside; the repro above is intended to make that a one-sitting job for someone
with the kernel in front of them.

## Workaround in use and its measured cost

Re-created the affected tags with `PARAMETER num_gpu N` where N = total layers − 1 (layer 0 on CPU), so `resolve_fused_ops` disables both fused GDN
paths on every load; the fused originals are kept for fresh-runner-only use. Measured on `qwen3.6:35b-a3b` (29-document sequence, one warm runner,
identical flags):

| config | prompt ms/tok | decode ms/tok | cold load | leak signature |
|---|---|---|---|---|
| fused ON (42/42) | 0.89 | 18.3 | 5.8 s (median of 2,104 loads) | DEFECT on a warm runner |
| **fused OFF (`num_gpu 41`, 41/42)** | **1.10–1.12 (+24 %)** | **23.0–23.7 (+26–29 %)** | 5.8 s | CLEAN ×2, request-exact |

Break-even for the alternative mitigation (keep the fused op, but a fresh runner per request) is ≈850 decode tokens per request, since each reload
costs the 5.8 s.

## Additional observations (same issue or a companion)

1. **Empty completions with the fused op disabled (0.32.14 bundle):** with a think-off prefill the model returns EOS as its *first* token on a
   prompt-dependent subset of free-text prompts (19/24 synthetic prompts, 8–1,057 tokens); never with the op enabled, never under a JSON grammar, and
   0/24 on the 0.34.1 bundle at `-ngl 41`. Reported as a companion observation, not as the same defect.
2. **0.34.1 with layer 0 on CPU still leaks 1 in 29** (deterministic, request 13 → 14 on our private document sequence) where 0.32.14 with the op
   disabled is 0/29 across seven runs, and 0.34.1 logs no GDN resolver lines while decoding at the fused speed — consistent with the GDN fusion having
   moved to a generic subgraph-fusion path that the layer-0 placement rule no longer gates.
3. **MTP speculative decoding** (`--spec-type draft-mtp`, `qwen3.8:27b`): the draft context's `resolve_fused_ops` enables the fused GDN op even when the
   target context has it disabled; the 3-request probe does not leak on this path even fully fused, so the draft state's effect is untested.

## Evidence in this repository

Every file below is byte-identical to what the probe wrote on our box; `SHA256SUMS` covers the whole tree (`sha256sum -c SHA256SUMS`).

| file | what |
|---|---|
| `rc3/leak_probe2.py`, `rc3/synth_docs.py` | the repro |
| `logs/rc3/synth2_qwen36_ngl42.json` | qwen3.6 fused ON — R2/R3 leak (`leak_from_A` set on both) |
| `logs/rc3/synth2_qwen36_ngl42.server.log` | verbose server log: resolver `enabled` ×2, `offloaded 42/42`, R2 `restored context checkpoint … n_past = 31`, R3 `cached n_tokens = 0, memory_seq_rm [0, end)` |
| `logs/rc3/synth2_qwen36_ngl42.proof.txt` | `/proc/<pid>/exe` + `.so` maps, resolver/offload/ubatch lines |
| `logs/rc3/synth2_qwen36_ngl41.{json,server.log,proof.txt}` | qwen3.6 fused OFF control — no leak; both GDN paths `set to disabled`, `offloaded 41/42` |
| `logs/rc3/synth2_qwen38_ngl66.{json,server.log,proof.txt}` | qwen3.8 fused ON — R2/R3 leak |
| `logs/rc3/synth2_qwen38_ngl65.{json,server.log,proof.txt}` | qwen3.8 fused OFF control — no leak |

The 0.34.1 observations above come from a 29-document sequence over private documents; those logs are not included.

## What "fixed" means to us

A build that passes `rc3/leak_probe2.py` with the fused op ON (all layers on GPU) twice on both models, plus a 29-document warm-runner sequence with
0 cross-document leaks — then our tags revert to the fused originals.
