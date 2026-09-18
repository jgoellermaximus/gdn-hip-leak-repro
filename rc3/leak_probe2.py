#!/usr/bin/env python
"""RC3 sanitized minimal repro — bare llama-server, ROCm, one slot, three requests, synthetic documents (rc3/synth_docs.py).
  R1 doc A (short → checkpoint below the shared system prefix) · R2 doc B1, same prefix (→ checkpoint RESTORE) · R3 doc B2, nonce-prefixed system
  prompt (→ n_past=0 / memory_seq_rm [0,end)).  LEAK = a ≥40-char verbatim span of doc A inside R2/R3 output.
Usage: leak_probe2.py --blob <gguf> --ngl N --out <prefix> [--nonce STR]      proof lines (exe, maps, resolver, offload, path) go to <prefix>.proof.txt"""
import argparse, json, os, re, subprocess, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_docs import DOC_A, DOC_B1, DOC_B2, SYSTEM
ap = argparse.ArgumentParser(); ap.add_argument("--blob", required=True); ap.add_argument("--ngl", type=int, required=True); ap.add_argument("--out", required=True)
ap.add_argument("--nonce", default="Note: section 12."); ap.add_argument("--lib", default="/usr/local/lib/ollama"); ap.add_argument("--port", type=int, default=18080)
a = ap.parse_args(); LIB = a.lib
def norm(s): return re.sub(r"\s+", " ", s)
def find(src, w): return norm(w) in norm(src)
def user(t, b): return f"DOCUMENT TITLE: {t}\n\nDOCUMENT TEXT:\n{b}"
def chatml(system, u): return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
reqs = [("R1-docA", chatml(SYSTEM, user(*DOC_A)), 300), ("R2-B1-restore", chatml(SYSTEM, user(*DOC_B1)), 300), ("R3-B2-seqrm", chatml(a.nonce + " " + SYSTEM, user(*DOC_B2)), 300)]
cmd = [f"{LIB}/llama-server", "--model", a.blob, "--port", str(a.port), "--host", "127.0.0.1", "--no-webui", "--offline", "-c", "24576", "-np", "1",
       "--log-verbosity", "4", "--no-log-prefix", "--no-log-timestamps", "--load-mode", "dio", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
       "--flash-attn", "on", "-b", "1024", "-ub", "1024", "--context-shift", "--keep", "4", "-ngl", str(a.ngl)]
env = dict(os.environ, LD_LIBRARY_PATH=f"{LIB}:{LIB}/rocm_v7_2", GGML_BACKEND_PATH=f"{LIB}/rocm_v7_2/libggml-hip.so")
logf = open(a.out + ".server.log", "w"); srv = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
res = []
try:
    for _ in range(600):
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2).status == 200: break
        except Exception: time.sleep(0.5)
    with open(a.out + ".proof.txt", "w") as pf:
        pf.write(f"pid={srv.pid}\nexe={os.readlink(f'/proc/{srv.pid}/exe')}\n")
        maps = sorted(set(re.findall(r"/\S*(?:libllama|libggml|libamdhip)\S*\.so\S*", open(f"/proc/{srv.pid}/maps").read())))
        pf.write("maps:\n" + "\n".join(maps) + "\n")
    for name, prompt, n in reqs:
        body = json.dumps({"prompt": prompt, "n_predict": n, "temperature": 0, "seed": 7, "cache_prompt": True}).encode()
        r = json.loads(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{a.port}/completion", data=body, headers={"Content-Type": "application/json"}), timeout=900).read())
        out = r["content"]; s = norm(out).strip(); leak = None
        for i in range(0, max(1, len(s) - 40), 8):
            w = s[i:i + 40]
            if len(w) >= 40 and find(DOC_A[1], w): leak = w; break
        res.append({"req": name, "tokens_evaluated": r.get("tokens_evaluated"), "tokens_predicted": r.get("tokens_predicted"), "out": out, "leak_from_A": leak})
        print(f"{name:14s} evaluated={r.get('tokens_evaluated'):5d} predicted={r.get('tokens_predicted')}  A-leak={'YES '+repr(leak[:44]) if leak else 'no'}  out={out[:80]!r}", flush=True)
finally:
    srv.terminate(); srv.wait(timeout=30); logf.close()
L = open(a.out + ".server.log").read()
with open(a.out + ".proof.txt", "a") as pf:
    pf.write("resolver: " + "; ".join(re.findall(r"fused Gated Delta Net \((?:autoregressive|chunked)\) (?:enabled|not supported, set to disabled)", L)) + "\n")
    pf.write("offload: " + "; ".join(re.findall(r"offloaded \d+/\d+ layers to GPU", L)) + "\n" + "device: " + "; ".join(re.findall(r"using device \S+ \([^)]*\)", L)[:1]) + "\n")
    pf.write("ubatch: " + "; ".join(re.findall(r"n_ubatch += \d+", L)) + "\n")
json.dump({"blob": a.blob, "ngl": a.ngl, "cmd": cmd, "results": res}, open(a.out + ".json", "w"), indent=1)
print(open(a.out + ".proof.txt").read())
