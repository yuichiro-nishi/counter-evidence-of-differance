# Counter-Evidence of Différance

**A Refutation from Engineering and Mathematics**

Yuichiro Nishi · Version 1.0.0 · 2026-04-22

---

## Abstract

Derrida's *différance* claims that meaning is never present — that it is always produced through differences and perpetually deferred. The claim is universal: it admits no domain in which meaning is exempt. This paper tests that universality against engineering and mathematics.

It fails.

In Transformer architecture, each token carries a determinate embedding before any contextual computation begins. The attention mechanism computes relevance through inner products of these vectors: an operation undefined unless both operands are determinate. At every forward pass, the softmax function compresses the full vocabulary into a probability distribution summing to exactly one. After training, model weights are frozen. Determinacy, once achieved, is irreversible.

These are not arguments. They are measurements — reproducible across three independently trained models (GPT-2, Pythia, TinyLlama), verified by scripts that any researcher can execute.

The refutation is not limited to Transformers. Shannon's channel coding theorem guarantees information recovery through noisy channels. Type systems determine meaning before execution. Mathematical structures are invariant across languages. The outside of the text exists.

This paper presents the first constructive refutation of *différance* grounded in reproducible engineering. Gunkel (2025) argued that large language models actualize *différance*; examining the complete pipeline rather than one layer, we reach the opposite conclusion.

> The play of differences is real. The indeterminacy is not.

---

## Repository Contents

| Path | Content |
|---|---|
| `paper/main.pdf` | **Main paper (English)** |
| `paper/main.tex` | LaTeX source |
| `scripts/` | Reproducible verification scripts |
| `results/` | Verification results (JSON; 3 models × 8 per-model tests + 2 cross-model tests) |
| `figures/` | Pipeline figure (PDF + LaTeX source) |
| `LICENSE` | CC BY 4.0 (paper, figures) |
| `LICENSE-CODE` | MIT (verification scripts) |
| `CITATION.cff` | Machine-readable citation metadata |

---

## Reproduction

All verification results in this repository can be independently reproduced.

### Stage 1 — Pure Python (zero dependencies)

Confirms the mathematical properties of the verification protocol without requiring any ML framework.

```bash
python scripts/verify_properties.py
```

### Stage 2 — Real models (PyTorch + transformers)

Verifies the same properties (plus additional cross-model tests) on three independently trained Transformer models:

* GPT-2 small (OpenAI, 124M parameters) — `openai-community/gpt2`
* Pythia 160M (EleutherAI, 162M parameters) — `EleutherAI/pythia-160m`
* TinyLlama 1.1B (TinyLlama, 1.1B parameters) — `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

```bash
pip install torch transformers scipy
python scripts/verify_real_models.py --models gpt2,pythia,tinyllama --device cpu
```

CPU execution is sufficient for GPT-2 small and Pythia 160M. TinyLlama requires ~4 GB of RAM. All models are open-weight and downloadable via the Hugging Face Hub without authentication. Approximately 2.5 GB disk space is needed for cached model weights.

Results are written to `results/`. All 26 verifications (3 models × 8 per-model tests + 2 cross-model tests) pass in the reference run.

### Per-model tests (T1–T8)

1. **T1 Embedding determinacy** — token embeddings are bitwise identical on repeated access; all values finite.
2. **T2 Inner-product determinacy** — ⟨e_a, e_b⟩ yields a finite scalar.
3. **T3 Softmax closure** — output probability distribution sums to one within float precision.
4. **T4 Attention simultaneity** — full n×n attention matrix produced in one step; each row sums to one.
5. **T5 Weight freezing at inference** — embedding weights bit-identical before and after 10 inference calls.
6. **T6 EOS termination** — P(EOS) > 0 throughout 32 generation steps; greedy generation halts under the model's own EOS emission where applicable.
7. **T7 Gradient descent + post-training freezing** — 200 AdamW steps reduce loss; post-training weights bit-identical across 10 eval forward passes.
8. **T8 Prompt determinacy** — 100 forward passes on a fixed prompt yield bit-identical logits.

### Cross-model tests (T9–T10)

9. **T9 Architecture diversity** — the tested set spans at least three distinct Transformer families (GPT-2, GPT-NeoX, Llama-2).
10. **T10 Cross-model embedding geometry** — pairwise cosine similarity of orthogonal-Procrustes-aligned embeddings exceeds the shuffled-row baseline for every model pair.

---

## Falsification Conditions

Per §9.2 of the paper, the refutation presented here would be overturned if one or more of the following were established:

1. **The Transformer counter-example fails** — if it can be shown that the determinacy of embeddings, the closure of the softmax distribution, the simultaneity of attention, the EOS termination mechanism, or the irreversibility of frozen weights is not in fact exhibited by the systems examined in §§2–3, the principal counter-example is withdrawn.
2. **The Transformer is not a sign system** — if a principled criterion places Transformer operation outside the scope of *signification* as Derrida used the term, and that criterion can be defended without restricting *différance*'s declared universality (the first horn of the trilemma in §1.1), the Transformer counter-example is disqualified on scope grounds.
3. **The independent counter-examples fail** — if Shannon's channel coding theorem can be shown not to guarantee signifier recovery, or if verified-compilation systems such as CompCert can be shown not to preserve meaning under the correspondence invoked in §4.2, the corresponding independent lines of counter-evidence are withdrawn.
4. **All independent lines are simultaneously disqualified** — because the refutation is triangulated, a single counter-example suffices. To overturn the refutation in its entirety, every independent line presented in §§2–5 must be disqualified.

The explicitness of these conditions contrasts with *différance*, which provides no conditions under which it would be falsified.

---

## Citation

If you reference this work:

```bibtex
@article{Nishi2026CounterEvidenceDifferance,
  title   = {Counter-Evidence of Différance: A Refutation from Engineering and Mathematics},
  author  = {Nishi, Yuichiro},
  year    = {2026},
  note    = {Version 1.0.0, released 2026-04-22},
  url     = {https://github.com/yuichiro-nishi/counter-evidence-of-differance}
}
```

GitHub users can also use the "Cite this repository" button (powered by `CITATION.cff`).

---

## License

* **Paper, figures**: [CC BY 4.0](./LICENSE)
* **Verification scripts** (`scripts/*.py`): [MIT](./LICENSE-CODE)

You are free to share and adapt this work for any purpose, including commercial use, provided appropriate credit is given.

---

## Contact

Yuichiro Nishi · `yuichiro@yuichironishi.com`

Independent researcher. Correspondence, corrections, and counter-counter-evidence are welcome.
