#!/usr/bin/env python3
"""Deterministic synthetic documents for the sanitized leak repro. No real names, files, or quotes. Doc A is short (target ≈1.05–1.15k prompt
tokens so the server's checkpoint lands below the shared system-prefix length); B1/B2 are long. Vocabulary is unique per doc so leaks are attributable."""
import random
def make(seed, name, subject, n_para, vocab):
    r = random.Random(seed); out = [f"# {name}\n", f"Subject: {subject}.\n"]
    for i in range(n_para):
        w = [r.choice(vocab) for _ in range(6)]
        out.append(f"Clause {i+1}. The {w[0]} committee resolved that the {w[1]} ledger shall record every {w[2]} event before the {w[3]} review, "
                   f"and that no {w[4]} transfer may precede a signed {w[5]} note. Reference code {r.randint(1000,9999)}-{r.choice('ABCDEF')}.\n")
    return "".join(out)
VA = ["harbor","lantern","quarry","saffron","meridian","cobalt","tallow","gable","sextant","bramble"]
VB1 = ["velvet","onyx","pumice","gantry","tundra","falcon","mortar","zephyr","tiller","cinder"]
VB2 = ["ember","sable","glacier","paddock","ironwood","monsoon","keel","lichen","rampart","tarn"]
DOC_A  = ("Harbor Lantern Charter", make(1, "Harbor Lantern Charter", "governance of the harbor lantern committee", 19, VA))
DOC_B1 = ("Velvet Gantry Ledger Rules", make(2, "Velvet Gantry Ledger Rules", "operating rules for the velvet gantry ledger", 80, VB1))
DOC_B2 = ("Ember Rampart Review Procedure", make(3, "Ember Rampart Review Procedure", "procedure for the ember rampart review", 64, VB2))
SYSTEM = ("You write extractive one-line abstracts of documents. Output JSON only with keys `abstract` (one sentence naming the document's concrete "
          "subjects and terms) and `quote` (a verbatim span of 40-300 characters copied exactly from the document).")
if __name__ == "__main__":
    for t, b in (DOC_A, DOC_B1, DOC_B2): print(t, len(b), "chars")
