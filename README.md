# JevFlash

A personal replication experiment inspired by **NanoJev** (itself a small
open replica of Typesafe AI's "Jev" system-one decision models). This repo
adapts that idea to run end-to-end on Apple Silicon (MPS) instead of CUDA,
and swaps in different backbones to see how much the choice of base model
actually matters for this architecture.

## What this is

Instead of generating text token-by-token, the model scores a fixed set of
candidate answers to a question about a given state, in a single backbone
forward pass:

```
input:  state + question + one candidate answer  (repeated per candidate)
output: one scalar score per candidate -> softmax -> probability distribution
```

Three question types are supported: `boolean` (yes/no), `choice` (pick one of
K candidates, K = 2..255), and `score` (an ordered level, e.g. severity 0-3).
`choice` questions get an extra small self-attention pass so candidates can be
scored relative to each other, not purely in isolation.

See [`jevflash/train.py`](jevflash/train.py) for the full implementation.

## What's different from upstream NanoJev

- **Device-agnostic**: runs on CUDA, Apple MPS, or CPU. Upstream NanoJev's
  scripts hardcode CUDA and refuse to run without it.
- **Gold-label training**: the toy dataset here has no teacher-model
  distillation targets, so training uses one-hot gold labels
  (`--objective gold`) instead of teacher-probability distillation.
- **Backbone comparison**: trained the same architecture on two backbones,
  [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) (the
  original NanoJev backbone) and
  [Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base), to see
  whether a bigger backbone actually helps on this task.

## Data

`jevflash/build_decisions.py` generates a small, self-authored, CC0-1.0
synthetic dataset (support-ticket routing / access-request questions) with
no external downloads required. Five splits: train, dev, calibration, test,
ood.

```bash
python jevflash/build_decisions.py --output-dir data/toy_v1
cat data/toy_v1/{train,dev,calibration,test,ood}.jsonl > data/toy_v1/all.jsonl
```

## Running a backbone

```bash
python jevflash/train.py \
  --input data/toy_v1/all.jsonl \
  --output-dir runs/<name> \
  --model Qwen/<backbone> \
  --device auto   # auto-picks cuda > mps > cpu
```

## Results

See [`results/`](results/) for per-run logs and `summary.json` outputs.
Comparison table below is filled in after both runs complete.

| Backbone | Params | Dev accuracy (best) | Dev gold-NLL (best) | Final eval accuracy | Training time |
|---|---|---|---|---|---|
| Qwen3-0.6B-Base | 0.6B | TBD | TBD | TBD | TBD |
| Qwen3-1.7B-Base | 1.7B | TBD | TBD | TBD | TBD |

## License

MIT (see [LICENSE](LICENSE)). `jevflash/train.py` and
`jevflash/build_decisions.py` are adapted from
[NanoJev](https://github.com/TianyuCodings/NanoJev) (MIT License) — see
[THIRD_PARTY_NOTICE_NanoJev_MIT.txt](THIRD_PARTY_NOTICE_NanoJev_MIT.txt).
