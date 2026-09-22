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
- **Backbone**: trained on
  [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base), the
  original NanoJev backbone. A [Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base)
  comparison run was attempted but abandoned partway through (see
  [Results](#results) below) — it's a reasonable next step if you have
  access to a CUDA GPU.

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

| Backbone | Params | Dev accuracy (best) | Dev gold-NLL (best) | Final eval accuracy | Training time |
|---|---|---|---|---|---|
| Qwen3-0.6B-Base | 0.6B | 100% | 3.5e-6 | 95.4% | 31.3 min (132 steps, CPU) |
| Qwen3-1.7B-Base | 1.7B | — | — | — | abandoned at step 37/132 |

Both runs were on CPU, not GPU — this Mac has no CUDA device, and PyTorch's
MPS (Apple GPU) backend leaked memory unboundedly on this workload (see
commit history): it JIT-compiles and caches a distinct graph per unique
input shape, and this trainer's batches vary in both token length and
candidate count every step, so the graph cache never stops growing
(observed 26GB+ resident before the run was killed).

The 0.6B run finished in ~31 minutes on CPU. The 1.7B run was expected to
take proportionally longer (~3x params) but instead ran at roughly **13x**
the per-step time of 0.6B — CPU throughput did not scale linearly with
model size on this hardware, likely a memory-bandwidth bottleneck rather
than a compute one. At that rate it would have needed 5+ hours, so it was
stopped at step 37/132 rather than let run. Worth rerunning on an actual
GPU (CUDA) if you want the real backbone comparison — `--device cuda` is
already supported in `train.py`.

Note the 0.6B numbers themselves should be read as a training-dynamics
sanity check, not a generalization result: this toy dataset has only 64
train / 16 dev states, so hitting 100% dev accuracy after ~130 steps is the
model memorizing a very small, templated dataset, not evidence of broad
capability.

## ViZDoom

The original Jev/NanoJev line of work evaluates on ViZDoom scenarios
(aiming/shooting), not just templated text. `jevflash/doom_env.py` is
NanoJev's headless ViZDoom adapter (unmodified); `build_doom_decisions.py`
runs the `basic` scenario (a stationary monster, strafe left/right and
shoot) under a simple proportional aim-and-shoot heuristic and records each
step as a `{state, question, gold_action}` row, using the heuristic's own
choice as the gold label (there's no learned "teacher" involved). 16 train
/ 4 dev / 4 calibration / 8 test / 4 ood episodes, 149 total decision
questions — the heuristic itself hits 100% episode success (it's simple:
strafe toward the target's screen-space center, shoot once aligned).

```bash
pip install -r requirements-vizdoom.txt   # vizdoom==1.3.0, gymnasium==1.3.0
python jevflash/build_doom_decisions.py --output-dir data/doom_basic_v1
cat data/doom_basic_v1/{train,dev,calibration,test,ood}.jsonl > data/doom_basic_v1/all.jsonl
python jevflash/train.py --input data/doom_basic_v1/all.jsonl \
  --output-dir runs/doom_basic_0.6b --model Qwen/Qwen3-0.6B-Base \
  --device cpu --batch-questions 6 --skip-native-baseline
```

**A real bug surfaced here**: ViZDoom states are longer text (~316 tokens
average vs. the toy dataset's ~72), and once the backbone unfreezes,
full fine-tuning retains every layer's activations for the whole sequence —
this blew up to 60GB+ resident memory and thrashed the machine on plain
CPU (not just MPS). Fixed with `model.backbone.gradient_checkpointing_enable()`
in `train.py`, which recomputes activations during backward instead of
storing them. Worth knowing if you extend this to longer/richer states.

### Closed-loop result (model actually playing, not just scoring a dataset)

`jevflash/play_doom.py` loads a trained checkpoint and drives live ViZDoom
episodes, picking the argmax action each step — a genuinely different test
than scoring the static dataset above.

| Policy | Success rate | Mean steps-to-kill (on success) | Timed out |
|---|---|---|---|
| Trained Qwen3-0.6B-Base (108 steps, offline eval 48.3% acc) | **25%** (5/20) | 7.6 | 15/20 |
| Uniform random action | **45%** (9/20) | 12.9 | 11/20 |

The trained model did *worse* than random action selection. This is a
genuine negative result, not a bug: the training set (110 questions from
16 episodes) came entirely from the heuristic's own tight, self-correcting
trajectory, so the model never saw what to do after a mistake. In live
play, once it drifts even slightly off that narrow expert path, errors
compound with no recovery signal in training — a textbook imitation-learning
distribution-shift failure (the kind DAgger-style methods exist to fix), on
top of a training set that's simply too small (110 examples) to fully
fine-tune a 0.6B model without overfitting to spurious per-episode details
in the raw JSON state text.

Run it yourself:
```bash
python jevflash/play_doom.py --checkpoint-dir runs/doom_basic_0.6b \
  --episodes 20 --device cpu --include-random-baseline \
  --output results/doom_basic_eval.json
```

### Attempt 2: DAgger data + frozen backbone (failed, kept for the record)

Diagnosis of attempt 1: the training data only contained states visited by
the heuristic's own clean trajectory, so the model never saw how to
recover from a mistake. Fix attempted: `jevflash/collect_dagger_doom.py`
lets the *trained* model drive its own live rollouts (16 episodes), labels
every state it visits with the heuristic's corrective action (not the
model's), and aggregates that with the original data — standard DAgger
(Ross et al., 2011). This took training data from 110 to 648 questions.
Separately, `train.py` gained `--freeze-backbone` (train only the decision
head, backbone never unfreezes) on the theory that fully fine-tuning a
0.6B model on this little data was itself the overfitting risk.

Result: **worse, not better — 0% success (0/20)**, every episode timed
out. Checked `predictions.jsonl` directly: the model predicts `"left"` for
**100% of all 149 test questions**, matching the offline "accuracy" of
43.6% purely because `left` happens to be the plurality gold label (65/149).
It learned zero state-dependent behavior; dev accuracy was identically
36.7% at every single eval checkpoint from step 12 through step 132 — a
constant-prediction plateau, not a training curve.

Takeaway: freezing the *entire* backbone was too aggressive for this task.
A frozen, generically-pretrained LM's hidden states apparently don't
linearly expose "target position relative to screen center" well enough
for a small head to extract without any backbone adaptation. The DAgger
data-aggregation idea itself is still sound — it just hasn't been tested
with an approach that can actually learn from it. Attempt 3 (below) reuses
the same full-fine-tuning setup that worked in attempt 1, on this larger
attempt-2 dataset, to isolate that variable.

### Attempt 3: DAgger data + full fine-tuning (in progress)

Same training approach as attempt 1 (full backbone fine-tuning, gradient
checkpointing) but on the attempt-2 combined dataset (648 train questions:
110 original heuristic + 538 DAgger-collected recovery states). Results
pending — see `runs/doom_basic_0.6b_dagger_full/` and this section will be
updated once it completes.

## License

MIT (see [LICENSE](LICENSE)). `jevflash/train.py` and
`jevflash/build_decisions.py` are adapted from
[NanoJev](https://github.com/TianyuCodings/NanoJev) (MIT License) — see
[THIRD_PARTY_NOTICE_NanoJev_MIT.txt](THIRD_PARTY_NOTICE_NanoJev_MIT.txt).
