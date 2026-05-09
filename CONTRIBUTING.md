# Contributing

Thanks for your interest in this project. This is primarily a research codebase, so contributions are welcome but the bar for changes that affect the training pipeline is high.

## Quick orientation

- **Simulation:** [`sim/envs/triple_pendulum_env.py`](sim/envs/triple_pendulum_env.py) — Gymnasium env, 16-dim obs, 1-dim continuous action.
- **Training:** [`training/train_m{2,3,4}_*.py`](training/) — one script per milestone, common MLflow + pipeline-notifier wiring.
- **Pipeline:** [`training/pipeline_notifier.py`](training/pipeline_notifier.py) and [`training/pipeline_stages.json`](training/pipeline_stages.json) — stage transitions consumed by the n8n orchestrator.
- **Launcher API:** [`scripts/launcher_api.py`](scripts/launcher_api.py) — HTTP service on the training host that n8n calls to start the next stage.
- **Wiki:** [the project wiki](https://github.com/fawraw/triple-pendulum-sim2real/wiki) is the long-form documentation. Keep page content there, not in `docs/`.

## Setting up a dev environment

```bash
git clone https://github.com/fawraw/triple-pendulum-sim2real.git
cd triple-pendulum-sim2real
./scripts/setup_env.sh              # creates .venv with all deps
source .venv/bin/activate
pytest                              # should pass
```

Headless rendering requires `MUJOCO_GL=osmesa` on Linux. On macOS no env var is needed.

## Running tests

```bash
pytest                              # all tests
pytest tests/test_env.py            # one module
pytest -k "test_obs_shape"          # one test
```

GitHub Actions runs the same `pytest` command plus an end-to-end smoke (200-step TQC training) on every PR.

## What kind of change am I making?

| Change type | Required |
|---|---|
| Bug fix in env / training | Tests covering the fix |
| New milestone (M5+) | Add config, training script, entry in `pipeline_stages.json`, whitelist in `launcher_api.py`, wiki page |
| Hyperparameter tuning | New YAML config; do not edit existing locked configs |
| Pipeline / n8n change | Update both the JSON workflow and the live n8n instance, document in wiki |
| Hardware (M6+) | BOM update + CAD/firmware in `hardware/` |

## Code style

- Type hints where they help: function signatures, public APIs.
- No comments that just restate the code. Reserve comments for **why** (constraints, tradeoffs, surprising behavior).
- One logical change per PR. Keep diffs reviewable.

## Hard rules

1. **Never commit secrets.** Check `git diff` for tokens, passwords, `.env` content. Pre-commit hook catches some, not all.
2. **MLflow callbacks must be resilient.** Wrap `mlflow.log_metric` in try/except — a network blip should not kill an 8-hour training run.
3. **Pipeline notifier secrets stay out of disk artifacts.** They go in the POST body only.
4. **Launcher whitelist is enforced.** New training modules must be added to `ALLOWED_MODULES` in [`scripts/launcher_api.py`](scripts/launcher_api.py).

## Releasing a milestone

When a milestone passes acceptance:

1. Tag the commit: `git tag -a v0.X -m "M3b complete"`
2. Update [Results.md in the wiki](https://github.com/fawraw/triple-pendulum-sim2real/wiki/Results) with per-EP / per-transition success rates.
3. Update the milestone status table in `README.md`.
4. Push the tag: `git push --tags`.

## Commit messages

Format: `[Type] short description in present tense`

Types: `[Add]`, `[Fix]`, `[Update]`, `[Refactor]`, `[Docs]`, `[Test]`, `[Pipeline]`, `[Hardware]`.

Body: explain the why if non-obvious. Reference an issue or MLflow run if relevant.

## Reporting security issues

If you find a credential exposed in the repo, an injection vulnerability in the launcher API, or any other security issue: open a private security advisory via GitHub instead of a public issue. Force-push history rewrites have been used in this repo before; we know how to do it again.
