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

### Attempt 3: DAgger data + full fine-tuning (also failed — new root cause found)

Same training approach as attempt 1 (full backbone fine-tuning, gradient
checkpointing) but on the attempt-2 combined dataset (648 train questions:
110 original heuristic + 538 DAgger-collected recovery states).

Result: **also 0% success (0/20) in closed-loop play**, and offline
accuracy identical to attempt 2's — 43.6%, because `predictions.jsonl`
shows the model again predicts `"left"` for **100% of all 149 test
questions**. Same degenerate collapse as attempt 2, despite the backbone
being fully unfrozen this time.

This rules out "frozen backbone" as the sole cause and points at something
more specific: **checkpoint selection**. The training log shows dev
gold-NLL bottoming out at **step 36 of 132** and getting *worse* every eval
after that (1.32 at step 84, 2.39 at step 108, 1.58 at step 132) — a
classic overfitting curve, just peaking unusually early — even though raw
training loss kept dropping through later steps (down to 0.24 at step
108). Since `train.py` always keeps the checkpoint with the best dev-NLL,
it saved the step-36 checkpoint: essentially still close to the untrained
head's initial constant-bias guess, before the model had done much real
learning. The later checkpoints, which trained loss suggests learned
*something* more, were never evaluated in closed-loop play at all.

A likely contributor: the dev split (30 questions) is unchanged from the
original heuristic-only data — it was never augmented with DAgger
states the way train was. So the selection metric optimizes for a
narrow slice of the state distribution, while train also has to fit the
messier, partly-off-policy DAgger states. A checkpoint that stays close
to "always predict the safe majority class" can look deceptively good on
that narrow dev set without having learned anything useful.

**Not yet tried** (paused here for direction rather than spending another
~2 hours unattended):
- Evaluate the **last** checkpoint (step 132) in closed-loop play instead
  of best-by-dev-NLL, since its training loss was competitive.
- Augment the **dev** split with DAgger-collected states too, not just
  train, so checkpoint selection reflects the full distribution the model
  actually needs to handle.
- Accept that 3 attempts is a reasonable place to pause and decide whether
  further automated iteration is worth it.

### Attempt 4: more DAgger data (dev-augmentation fix was broken, but it still worked) — **first real improvement**

Intent: collect a second batch of DAgger states specifically to augment
the **dev** split (not just train), so checkpoint selection would reflect
the harder distribution instead of a narrow 30-question heuristic-only
dev set.

**Bug found mid-run**: `collect_dagger_doom.py` hardcoded `split: "train"`
on every row it wrote, regardless of output filename — `train.py` groups
rows by that internal field, not by which file they came from. So the 160
rows intended for `dev` silently landed in `train` instead. This attempt
ended up testing something different than intended: full fine-tuning with
more train data (808 questions total) but the *same* narrow 30-question
dev set as attempts 1-3. (Fixed now: `collect_dagger_doom.py` takes an
explicit `--split` argument for any future attempt at the real fix.)

Despite testing the wrong thing, the result is the best so far:

| Attempt | Backbone state | Offline accuracy | Predictions | Closed-loop success | vs. random (45%) |
|---|---|---|---|---|---|
| 1 | full fine-tune, 110 train Q | 48.3% | differentiated | 25% | worse |
| 2 | frozen, 648 train Q | 43.6% | 100% "left" (degenerate) | 0% | much worse |
| 3 | full fine-tune, 648 train Q | 43.6% | 100% "left" (degenerate) | 0% | much worse |
| 4 | full fine-tune, 808 train Q | 33.6% | "shoot" 106x / "left" 43x (2 classes, no "right") | **50%** | **better** |

Best checkpoint was step 84/132 — much later in training than attempts
2-3's step 36, and not degenerate this time (uses 2 of 3 action classes,
just miscalibrated toward over-shooting and never turning right). Lower
*offline* accuracy than attempts 2/3 but dramatically better *closed-loop*
play — a reminder that offline accuracy on this dataset is a poor proxy
for actual game performance, since a constant-predictor can look
deceptively decent on paper while a genuinely-reactive-but-imperfect
policy can look worse on paper and play much better live.

Takeaway: raw amount of training data mattered more here than precisely
which checkpoint-selection fix was applied. The properly-fixed
dev-augmentation experiment (now that `--split` actually works) is a
reasonable next step to see if it improves further on this 50% baseline.

**Correction, found by actually watching it play:** the 50% success rate
above is not real skill. Rendering live episodes with
`jevflash/render_model_episode.py` (captures actual screen frames while
the checkpoint drives the game, instead of just reading aggregate stats)
shows the model produces the **exact same 9-action cycle** —
`left, shoot, shoot, left, left, left, left, shoot, shoot` — repeating,
regardless of seed. Verified across 3 different seeds (different random
monster positions): byte-for-byte identical action sequences every time.
It is not reacting to the game state at all.

Root cause: the state JSON includes `remaining_decisions` and
`episode_tick` fields that count down/up **deterministically by a fixed
amount every step, in every episode, independent of where the target
actually is**. That's a far easier pattern to memorize ("do X when
remaining_decisions=38") than the real signal (target bbox position
relative to screen center) — classic feature leakage. The 50% success
rate is just whatever fraction of random monster starting positions
happen to align with this fixed, unconditional dance, not the model
tracking anything. The earlier "prediction diversity" check (2 classes
used across 149 offline test questions) didn't catch this, because it
only checks isolated random states, not whether behavior is actually
conditioned on them across a sequence — this is exactly why closed-loop,
watched play matters more than any offline metric here.

**Real next step**: strip or coarsely bucket `remaining_decisions` /
`episode_tick` from the state text before the next attempt, so the model
can't shortcut on step-index and is forced to use the target's actual
position.

### Attempt 5: leak fixed — first genuinely real result (still worse than random)

`jevflash/doom_env.py` now excludes `remaining_decisions`,
`remaining_ticks`, and `episode_tick` from the model-visible state.
Dataset regenerated from scratch (`data/doom_basic_v2/`, since the old
data has the leak baked into its serialized text) and retrained fresh
with the same full-fine-tuning config as attempt 1.

**Verified properly this time, before trusting any number**: rendered 3
different seeds with `render_model_episode.py` and compared the actual
action sequences directly.

| Seed | Actions | Outcome |
|---|---|---|
| 9000 | `right, right, shoot, right, shoot×36...` | timeout |
| 9005 | `right, right, shoot, right, shoot×7` | killed at step 11 |
| 9010 | `right, shoot, right, shoot×37...` | timeout |

Unlike attempt 4 (byte-for-byte identical regardless of seed), these
genuinely differ — different initial strafe counts, different outcomes.
The leak fix worked: **this is a real, state-conditioned policy, not a
memorized shortcut.**

But it has a real flaw: it strafes briefly, then locks into shooting
repeatedly for the rest of the episode regardless of whether it's still
aligned — it never re-adjusts after an initial miss. Full closed-loop
result (20 episodes, same seeds 9000-9019 as all prior attempts):

| Metric | Value |
|---|---|
| Success rate | **35%** (7/20) |
| Steps-to-kill on success | 4, 4, 7, 7, 11, 11, 13 (genuinely varied) |
| Outcomes | 7 killed, 13 timed out |
| vs. random (45%) | worse |
| vs. attempt 1 (25%, also real) | better |

Full attempt comparison:

| Attempt | Real or shortcut? | Closed-loop success |
|---|---|---|
| 1 | real | 25% |
| 2 | degenerate (constant "left") | 0% |
| 3 | degenerate (constant "left") | 0% |
| 4 | **fake** (fixed action-index shortcut, exposed by rendering) | 50% (not real) |
| 5 | **real** (verified via multi-seed action diversity) | 35% |

Takeaway: fixing the leak produced the first checkpoint that's
simultaneously non-degenerate *and* passes the "does behavior actually
vary with the state" check — genuine progress on validity, even though
raw performance (35%) is still below random and below attempt 4's fake
number. The remaining problem is a real policy-quality issue (no
re-alignment after the first shot), not a data-leakage one. A next step
worth trying: reward/penalize based on whether shots actually land, or
add training examples specifically covering "you just missed, realign"
states.

## License

MIT (see [LICENSE](LICENSE)). `jevflash/train.py` and
`jevflash/build_decisions.py` are adapted from
[NanoJev](https://github.com/TianyuCodings/NanoJev) (MIT License) — see
[THIRD_PARTY_NOTICE_NanoJev_MIT.txt](THIRD_PARTY_NOTICE_NanoJev_MIT.txt).
