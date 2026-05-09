# Runbook — Triple Pendulum Pipeline

Operational reference for the running training pipeline. **Read top-down when something breaks.**

## Quick status

```bash
# All-in-one: launcher + active sessions + last MLflow run
./scripts/tp_status.sh

# Individual checks:
curl -s http://10.1.4.232:8765/status | jq                  # active training sessions
curl -s http://10.1.4.230:5000/api/2.0/mlflow/runs/search \
  -H "Content-Type: application/json" \
  -d '{"experiment_ids":["1","2"],"max_results":5,"order_by":["start_time DESC"]}' | jq '.runs[] | {run_id: .info.run_id, status: .info.status, name: .info.run_name}'
```

## Hosts

| Service | CT | IP | Port |
|---|---|---|---|
| n8n | 1003 | `10.1.4.226` | 5678 |
| MLflow | 1016 | `10.1.4.230` | 5000 |
| Grafana / InfluxDB | 1017 | `10.1.4.231` | 3000 / 8086 |
| Training + launcher | 1018 | `10.1.4.232` | 8765 |

`https://n8n.faji.co` (NPM) is currently broken for the n8n vhost (TLS handshake fail). Use the HTTP-direct URL `http://10.1.4.226:5678/...` until the NPM SSL config is fixed.

## Common failures

### 1. Training stopped, no notification fired

**Symptom:** MLflow run shows FINISHED, no Telegram, no n8n auto-launch.

**Diagnose:**
```bash
ssh root@10.1.4.232 'tail -20 /var/log/tp-launcher/<session>.log'
# Look for: "n8n webhook: HTTP <code>" or "n8n webhook FAILED after N attempts"
```

**Fix:** `pipeline_notifier` retries 3× (1s, 4s, 16s) before giving up. If all retries fail and `TELEGRAM_FALLBACK_BOT_TOKEN` is set in `/etc/triple-pendulum/launcher.env`, you should still receive a Telegram. Otherwise, manually trigger:
```bash
RESULT=$(ssh root@10.1.4.232 cat /opt/triple-pendulum/repo/results/<run_name>.json)
PAYLOAD=$(echo "$RESULT" | jq '. + {pipeline_secret: "<from launcher.env>"}')
curl -s -X POST -H "Content-Type: application/json" \
  -d "$PAYLOAD" http://10.1.4.226:5678/webhook/triple-pendulum-pipeline
```

### 2. n8n execution failed with `Module 'X' is disallowed`

**n8n's vm2 sandbox blocks several Node built-ins.** Confirmed blocked: `crypto`, `child_process`, `fs`, `os`, `dgram`, `net`, `tls`, `vm`, `worker_threads`. Allowed: `http`, `https`, `path`, `url`, `querystring`.

**Fix:** rewrite the Code node without `require()`. For timing-safe string comparison, use the pure-JS `safeEqual()` shim (see `n8n/triple_pendulum_pipeline.json`).

### 3. n8n webhook returns 200 but workflow errored

`responseMode: onReceived` returns 200 immediately; errors are logged in the executions list:
```bash
N8N_API_KEY="..."
curl -s -H "X-N8N-API-KEY: $N8N_API_KEY" \
  "http://n8n.faji.co/api/v1/executions?workflowId=WwwkbHr5hu2Xeiwy&limit=5" \
  | jq '.data[] | {id, status, startedAt}'

# Drill into one execution:
curl -s -H "X-N8N-API-KEY: $N8N_API_KEY" \
  "http://n8n.faji.co/api/v1/executions/<id>?includeData=true" \
  | jq '.data.resultData.error'
```

### 4. Launcher won't start after secret rotation

`scripts/launcher_api.py` refuses to start if `LAUNCHER_SECRET` is unset or matches the placeholder `YOUR_LAUNCHER_SECRET`. Systemd retries 5× in 300 s then gives up.

**Fix:**
```bash
ssh root@10.1.4.232 'cat /etc/triple-pendulum/launcher.env'   # verify secret set
ssh root@10.1.4.232 'systemctl status tp-launcher'
ssh root@10.1.4.232 'journalctl -u tp-launcher -n 50'
ssh root@10.1.4.232 'systemctl reset-failed tp-launcher && systemctl start tp-launcher'
```

### 5. `systemctl restart tp-launcher` killed my training

Was a real bug pre-fix. The unit now ships with `KillMode=process` (in `docs/launcher_api.service`). If you encounter this on an outdated CT 1018:
```bash
ssh root@10.1.4.232 'grep KillMode /etc/systemd/system/tp-launcher.service'
# If absent:
ssh root@10.1.4.232 "sed -i '/^Restart=/i KillMode=process' /etc/systemd/system/tp-launcher.service && systemctl daemon-reload"
# Do NOT restart while training is active.
```

### 6. MLflow run stuck as RUNNING after crash

When a training process crashes (OOM, network blip propagated, etc.) MLflow doesn't auto-mark the run as FAILED — the run sits as RUNNING forever. Manually fix:

```bash
RUN_ID="<id>"
curl -s -X POST -H "Content-Type: application/json" \
  http://10.1.4.230:5000/api/2.0/mlflow/runs/update \
  -d "{\"run_id\":\"$RUN_ID\",\"status\":\"FAILED\",\"end_time\":$(date +%s%3N)}"
```

A nightly janitor cron is in the backlog (audit M4).

### 7. Two trainings running simultaneously

Should not happen — the launcher checks via `pgrep` for any running `python -m training.train_m*` and 409s the second request. If you somehow end up with two:

```bash
ssh root@10.1.4.232 'tmux list-sessions; ps -ef | grep "training\.train_m" | grep -v grep'
ssh root@10.1.4.232 'tmux kill-session -t <name>'   # kill the duplicate
```

The duplicate is usually the older one if the new launch was initiated by n8n auto-pipeline; verify via tmux session start time.

### 8. Disk full on CT 1018 mid-run

```bash
ssh root@10.1.4.232 'df -h /opt/triple-pendulum'
# Top culprits: /opt/triple-pendulum/repo/checkpoints/ (multi-GB per run)
ssh root@10.1.4.232 'du -sh /opt/triple-pendulum/repo/checkpoints/*'

# Cleanup old runs (keep last 3):
ssh root@10.1.4.232 'cd /opt/triple-pendulum/repo/checkpoints && \
  ls -1tr | head -n -3 | xargs -r rm -rf'
```

## Manual operations

### Launch a training stage manually

```bash
LAUNCHER_SECRET="<from launcher.env>"
curl -s -X POST -H "Content-Type: application/json" \
  -d "{\"secret\":\"$LAUNCHER_SECRET\",\"module\":\"training.train_m3_all_eps\",\"config\":\"training/configs/m3c_all_eps_tqc.yaml\"}" \
  http://10.1.4.232:8765/launch
```

Allowed modules: `training.train_m2_upright`, `training.train_m3_all_eps`, `training.train_m4_transitions`.

### Auto-launch M4 after M3 passes

The pipeline sets `next.stage = 'M4'` but the M4 training script has a **cold-start guard**: it refuses to start if `pretrained_policy: null` in the config and `allow_cold_start` is not `true`. Before allowing the auto-launch, edit:
```yaml
# training/configs/m4_transitions_tqc.yaml
pretrained_policy: checkpoints/m3_all_eps_<run_name>/best_model.zip
```
Commit + push + pull on CT 1018.

### Rotate a secret

| Secret | Live source | Rotate |
|---|---|---|
| `LAUNCHER_SECRET` | `/etc/triple-pendulum/launcher.env` on CT 1018 | edit env file → `systemctl restart tp-launcher` (will NOT kill running training thanks to `KillMode=process`); update n8n Code node `LAUNCHER_SECRET` constant |
| `N8N_PIPELINE_SECRET` | same env file (consumed by training scripts), AND n8n Code node `PIPELINE_SECRET` | rotate both atomically; mismatch → all training notifications rejected |
| Telegram bot token | n8n Code node `TELEGRAM_BOT_URL`; optional `TELEGRAM_FALLBACK_BOT_TOKEN` in `launcher.env` | revoke via @BotFather, get new token, update both places |

After updating n8n, also re-sync via the script in `scripts/sync_n8n_workflow.py` (TODO: write this script — currently a manual `python3 -m` block in this repo's history).

### Check a single training's progress

```bash
RUN_ID="<from MLflow UI>"
for metric in timesteps rollout_ep_rew_mean; do
  echo "=== $metric ==="
  curl -s "http://10.1.4.230:5000/api/2.0/mlflow/metrics/get-history?run_id=$RUN_ID&metric_key=$metric" \
    | jq '.metrics[-1] // "no data"'
done
```

## Cloud GPU (RunPod)

Local CT 1018 is CPU-only. For M3c (4M steps, ~16h on CT) and M4 (5M steps,
~24h on CT) move to a GPU pod — same cost as CT but 5-10× wall-clock.

See [`runpod/README.md`](../runpod/README.md) for full setup.

Quick path once template is configured:

```bash
runpodctl create pod \
    --templateId "<id>" \
    --gpuTypeId "NVIDIA_RTX_A5000" \
    --env "TP_STAGE_MODULE=training.train_m4_transitions" \
    --env "TP_STAGE_CONFIG=training/configs/m4_transitions_tqc.yaml"
```

The pod auto-shuts down on training completion (`TP_AUTO_SHUTDOWN=1` default).

## Known operational gaps (backlog)

| Gap | Severity | Plan |
|---|---|---|
| n8n NPM SSL handshake | medium | debug NPM SSL cert / SNI for `n8n.faji.co`; until then HTTP-direct works |
| Static IP for CT 1018 | low | add DHCP reservation in UDM-ENT |
| Tracked binary blobs (assets/*.mp4, *.png) | low | migrate to git-lfs before they bloat history |
| MLflow zombie run janitor | low | add nightly cron on CT 1016 |
| Telegram chat_id hardcoded in n8n | low | move to `$env.TELEGRAM_CHAT_ID` |
| Embedded secrets in n8n Code node | medium | move to `$env.PIPELINE_SECRET` etc. |

## Secrets hygiene (operator checklist)

- `HISTCONTROL=ignorespace` in your shell, prepend curl with secrets with a leading space
- Or: source `~/.tp-pipeline-creds` and use `--data @-` from a heredoc
- Pre-commit hook at `.githooks/pre-commit` blocks known patterns; activate with `git config core.hooksPath .githooks`
