# RunPod / Cloud GPU training

Run training stages on a paid GPU pod instead of CT 1018 (CPU-only).

## When to use

- M3c (4M steps) finishes in ~1h on A5000 vs ~16h on CT 1018.
- M4 (5M steps) finishes in ~1.5h vs ~24h.
- M5 (domain randomization, 10M steps) becomes practical.

## One-time setup

1. Create a RunPod account at https://runpod.io (or use an existing one).
2. Add a payment method (most use cases stay under $30/month).
3. Create a **Network Volume** (50 GB suggested) so `mlruns/`, `checkpoints/`,
   and `results/` persist across pods. Cost: ~$5/month.
4. Generate a RunPod API key in Settings → API. Save to `credentials/`.
5. Build and push the Docker image (or use the public one when published):
   ```bash
   cd triple-pendulum-sim2real
   docker build -t fawraw/triple-pendulum:cuda121 -f runpod/Dockerfile .
   docker push fawraw/triple-pendulum:cuda121
   ```
6. Create a RunPod **Template**:
   - Container image: `fawraw/triple-pendulum:cuda121`
   - Container start command: `bash /workspace/triple-pendulum-sim2real/scripts/runpod_bootstrap.sh`
   - Volume mount path: `/workspace`
   - Environment variables: see below.
   - Expose ports: none (pod talks out, not in).

### Required env vars (in template or per-pod)

| Var | Example | Notes |
|---|---|---|
| `TP_STAGE_MODULE` | `training.train_m4_transitions` | which trainer to run |
| `TP_STAGE_CONFIG` | `training/configs/m4_transitions_tqc.yaml` | config path |
| `RUNPOD_API_KEY` | `<secret>` | required for auto-shutdown + idle watchdog |

### Recommended for cloud pods (notifications)

The pod cannot reach the LAN-only n8n at `10.1.4.226:5678`, so the
pipeline notifier's webhook will fail. Without a fallback, end-of-training
is silent. Set:

| Var | Notes |
|---|---|
| `TELEGRAM_FALLBACK_BOT_TOKEN` | bot token from @BotFather; pipeline_notifier POSTs a summary directly to Telegram when n8n is unreachable |
| `TELEGRAM_FALLBACK_CHAT_ID` | chat id (`@username` or numeric) |

### Optional env vars

| Var | Default | Notes |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `file:/workspace/mlruns` | network volume → survives pod death. Override with `http://...:5000` if you tunnel through Tailscale |
| `N8N_PIPELINE_WEBHOOK` | unset | only useful if you expose n8n publicly |
| `N8N_PIPELINE_SECRET` | unset | required if webhook is set |
| `TP_AUTO_SHUTDOWN` | `1` | set to `0` to keep pod alive (debug); idle watchdog still applies |
| `TP_IDLE_SHUTDOWN_MIN` | `30` | minutes of GPU<5% before forced podStop (cost guard) |
| `TP_REPO_REF` | unset (uses branch HEAD) | commit SHA or tag to pin training to a specific code version |
| `TP_REPO_BRANCH` | `main` | git branch to clone/reset to |

`RUNPOD_POD_ID` is injected automatically by RunPod into the container env.

### Recommended GPU types

| GPU | $/hr (community / on-demand) | M3c run | M4 run |
|---|---|---|---|
| RTX A4000 (16GB) | $0.20 / $0.30 | ~1.5h | ~2h |
| RTX A5000 (24GB) | $0.30 / $0.50 | ~1h | ~1.5h |
| RTX A6000 (48GB) | $0.60 / $0.80 | ~45min | ~1h |
| L40S (48GB) | $0.80 / $1.20 | ~30min | ~45min |
| H100 (80GB) | $2.00 / $3.50 | ~20min | ~30min |

For TQC on a [512,512] network, **A4000 or A5000 is the sweet spot** — anything bigger is GPU-underutilized and you pay for VRAM you don't need.

## Launch a run

### Via RunPod web UI

1. Pods → Deploy → choose your template
2. Pick GPU (A5000 community recommended for cost)
3. Override env vars if needed (different stage / config)
4. Deploy
5. Pod auto-shuts down on training completion

### Via RunPod CLI (`runpodctl`)

```bash
# Install once: https://github.com/runpod/runpodctl
runpodctl config --apiKey "<key>"

# Launch
runpodctl create pod \
    --templateId "<your-template-id>" \
    --gpuTypeId "NVIDIA_RTX_A5000" \
    --env "TP_STAGE_MODULE=training.train_m4_transitions" \
    --env "TP_STAGE_CONFIG=training/configs/m4_transitions_tqc.yaml"
```

### From your laptop, via launcher API on CT 1018

Currently the launcher API only spawns local tmux. Phase 3 (TBD) will add a
"runpod" backend so n8n can dispatch directly to a pod. For now, manually
launch via the web UI when M3 passes and HUMAN_REVIEW fires for M4.

## Monitoring a running pod

- RunPod web UI shows container logs in real time
- MLflow run is logged to your `MLFLOW_TRACKING_URI` if reachable
- Or `ssh root@<pod-ip>` (RunPod gives an SSH command per pod) and `tail -f /workspace/training.log`

## Costs and limits

- Network volume: $0.10/GB/month — keep small (under 100 GB)
- A5000 community: ~$0.30/hr — typical M3 run = $0.30, M4 = $0.45
- Spot pricing 30-60% cheaper but pre-emption risk; not recommended for
  multi-hour runs unless you've added checkpointing pickup logic
- Set a hard `$/month` budget alert in RunPod Settings → Billing

## Reverse-direction concerns

- The pod cannot reach `10.1.4.230:5000` (MLflow on CT 1016) unless you
  expose it via NPM or VPN (Tailscale recommended). Without it, the pod
  logs to local `mlruns/` on the network volume — you sync to MLflow CT
  later via `mlflow runs export`.
- Same for the n8n webhook. Either expose `n8n.faji.co` (currently broken
  TLS) or use a Tailscale endpoint.

See [docs/runbook.md](../docs/runbook.md) for general operational guidance.
