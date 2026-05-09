# Results

Each completed training run writes a JSON snapshot here via
`training/pipeline_notifier.py`. The snapshot contains:

- `milestone` — stage name (e.g. `M3b`)
- `run_name`, `run_id` — MLflow identifiers
- `timestamp` — UTC ISO-8601
- `config` — path to the YAML config used
- `metrics` — per-EP / per-transition success rates and means

The `pipeline_secret` field is **stripped** before writing to disk (only
included in the POST body to the n8n webhook).

These files are committed to git as a permanent record of every run. If you
need raw per-step metrics or the model artifact, follow the `run_id` to MLflow
at `http://10.1.4.230:5000`.
