# CLAUDE.md — JevFlash

## Git identity (check this before every commit/push)

This repo is a personal project published under the **themaker00001**
GitHub account, not the user's work identity. The user's *global* git
config is set to their work identity (name "Vaibhav Kathait", work
email) — that is correct for their other repos and must not be changed.

This repo has its own **local** git config override instead:

```
user.name  = themaker00001
user.email = the.maker00001@gmail.com
```

Before creating any commit in this repo, verify the local override is
still in place and takes effect:

```bash
git config user.name   # must print: themaker00001
git config user.email  # must print: the.maker00001@gmail.com
```

If either is missing or wrong (e.g. after a fresh clone, or if something
reset `.git/config`), fix it with:

```bash
git config --local user.name "themaker00001"
git config --local user.email "the.maker00001@gmail.com"
```

Do **not** set this globally (`--global`) — that would incorrectly change
the user's identity on their other, unrelated repositories.

Hugging Face: the account used for this project is `SUPER321` (full name
on file: "vaibhav kathait"), tied to `the.maker00001@gmail.com` — same
personal identity as the GitHub account above, different username. The
`HF_TOKEN` lives in this project's `.env` (gitignored, never commit or
print it).

## Project context

JevFlash is a from-scratch replica of NanoJev's non-generative decision-
model architecture (score every candidate action in one forward pass, no
text generation), fine-tuned on Qwen3-0.6B-Base and tested on ViZDoom's
"basic" scenario. Full experiment history (10 training attempts, what
failed and why, what finally worked) is in README.md — read it before
starting new experiments so you don't repeat an already-diagnosed dead
end (e.g. constant-prediction collapse tied to a specific random seed,
documented under "Attempt 10").
