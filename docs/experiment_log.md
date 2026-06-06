# Experiment Log

Only results from commands that actually ran are recorded here. Sections marked
*pending* have not been run yet and must not be filled with estimated numbers.

---

## Experiment 1 — tiny_transformer on 4 Gutenberg books

**Date:** 2026-06-06

### Data preparation (actually run, local)
Commands:
```bash
uv run python scripts/download_gutenberg.py --books alice holmes grimm poe --output_dir data/raw
uv run python scripts/prepare_corpus.py --input_dir data/raw --output_path data/processed/corpus.txt
```

Downloaded (raw character counts reported by the downloader):

| Book   | Work                                         | Raw chars |
|--------|----------------------------------------------|-----------|
| alice  | Alice's Adventures in Wonderland (Carroll)   | 167,712   |
| holmes | The Adventures of Sherlock Holmes (Doyle)    | 593,911   |
| grimm  | Grimms' Fairy Tales (Brothers Grimm)         | 549,776   |
| poe    | The Works of Edgar Allan Poe, Vol. 1         | 600,521   |

Processed corpus (also includes the tiny committed `sample_corpus.txt`):

- **Corpus length:** 1,831,344 characters
- **Vocabulary size:** 115 (114 unique characters + 1 `<unk>`)
- **Train / val split (90/10):** ~1,648,209 / ~183,135 characters

This is a reasonable size for the tiny model, so no "corpus too small" warning
applies.

### Training configuration
From `configs/tiny_transformer.yaml`, architecture unchanged. The only change was
`max_iters`, raised from 3000 to 8000 via `--max_iters_override 8000` (iteration
count only — not an architecture change), since a GPU made the longer run cheap.

- seed 42, batch_size 64, block_size 128
- n_embd 128, n_head 4, n_layer 4, dropout 0.2
- learning_rate 3e-4, max_iters 8000
- eval_interval 300, eval_iters 100
- Model size: **839,168 parameters** (reported at startup for vocab_size 115)

### Training run (actually run, Google Colab GPU)
Command:
```bash
python -m src.train --config configs/tiny_transformer.yaml \
    --corpus_path data/processed/corpus.txt --max_iters_override 8000
```

Results:

- **Device:** cuda (Google Colab GPU)
- **Parameters:** 839,168
- **Initial loss (iter 0):** train 4.7602 / val 4.7584
- **Final loss (iter 8000):** train 1.3049 / val 1.4472
- **Best recorded val loss:** 1.3911 (iter 7800; train 1.2316)
- **Loss curve:** `outputs/metrics/loss_curve.png`
- **Full metrics:** `outputs/metrics/losses.csv` (29 logged points)
- Wall-clock time: not recorded.

Observations (honest):

- Validation loss fell from ~4.76 to ~1.39–1.45 and was still slowly trending down
  at iter 8000, so a bit more training could help marginally.
- The eval estimates oscillate (e.g. 1.39 vs 1.49 between adjacent evals) because
  each evaluation averages random batches; this is estimation noise, not divergence.
- Train loss sits roughly 0.06–0.12 below val loss — a small, expected gap. No
  severe overfitting on this ~1.8M-character corpus.

### Generated sample (actually generated)
Generated locally from the downloaded checkpoint:
```bash
uv run python -m src.generate --checkpoint outputs/checkpoints/ckpt.pt \
    --prompt "The " --max_new_tokens 500 --temperature 0.8 --top_k 40
```

Excerpt (verbatim, not cherry-picked for quality):

> The Gryphon was with the later bride and was recil to the guest into the struck
> part of which we had much done to do be quite the side of a knee into the play my
> lady. The Little was already comployed some powerds. [...] Then they were staying
> the carry now. The morning beautiful of the little struggling his about before,
> and very spring at the rage of blue days of a cartrial.

Honest assessment: the model produces **locally plausible** text — correctly spelled
common words, capitalization, punctuation, quotation marks, and paragraph breaks,
in a Gutenberg-ish register — but it is **not globally coherent** (sentences do not
carry consistent meaning, and some tokens are non-words like "comployed"). This is
the expected ceiling for a ~0.84M-parameter character-level model and matches the
project's stated goal (clean implementation over text quality).

### Notes
- Trained from scratch; no pretrained weights.
- See `docs/colab.md` for the GPU runbook.
