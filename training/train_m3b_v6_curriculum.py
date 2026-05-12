"""M3b-v6: targeted fine-tuning curriculum from an existing baseline.

Unlike M3b-v5 (which started from scratch), v6 warm-starts from the best
existing 67.5% checkpoint and adds EP4/EP6 mastery via two short phases:

  Phase 1 — hardfocus  (600K steps, hard_ep_weight=20, LR=1e-4):
    Heavy EP4/EP6 exposure (~46% each).  Probe data shows 200K EP4-only steps
    → EP4=60%, so 276K-equivalent should hit similar.  LR/3 limits forgetting.

  Phase 2 — consolidate (500K steps, hard_ep_weight=2.5, LR=5e-5):
    Re-balance across all 8 EPs to recover any other-EP skills lost in phase 1.
    Tiny LR protects both phase-1 gains and prior skills.

Run from project root:

    MUJOCO_GL=osmesa python -m training.train_m3b_v6_curriculum
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.train_m3_all_eps import main as train_main  # noqa: E402


def run_phase(cfg_path: str, override_load_path: str = "") -> str:
    """Run one training phase. If override_load_path is non-empty, it replaces
    the load_model_path in the config (used for phase 2 to load phase 1's output)."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    if override_load_path:
        cfg["load_model_path"] = override_load_path
        tmp_path = Path(cfg_path).with_suffix(".tmp.yaml")
        with open(tmp_path, "w") as f:
            yaml.dump(cfg, f)
        cfg_to_use = str(tmp_path)
    else:
        cfg_to_use = cfg_path

    print(f"\n{'='*60}")
    print(f"Starting: {cfg.get('stage', cfg_path)}")
    load = cfg.get("load_model_path", "")
    if load:
        print(f"  warm-start from: {load}")
    print(f"{'='*60}\n")

    before = set((ROOT / "checkpoints").glob("*/final.zip"))
    train_main(cfg_to_use)
    after = set((ROOT / "checkpoints").glob("*/final.zip"))

    if override_load_path:
        try:
            Path(cfg_to_use).unlink()
        except Exception:
            pass

    new_finals = after - before
    if not new_finals:
        raise RuntimeError("Phase finished but no new final.zip found in checkpoints/")
    latest = max(new_finals, key=lambda p: p.stat().st_mtime)
    print(f"\n[curriculum] Phase checkpoint: {latest}")
    return str(latest)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", "--phase1_config", dest="phase1_config",
                   default="training/configs/m3b_v6_phase1_hardfocus.yaml")
    p.add_argument("--phase2_config",
                   default="training/configs/m3b_v6_phase2_consolidate.yaml")
    args = p.parse_args()

    t0 = time.time()
    print("[curriculum] === M3b-v6 targeted fine-tuning curriculum ===")
    phase1_ckpt = run_phase(args.phase1_config)
    print(f"[curriculum] Phase 1 complete in {(time.time()-t0)/3600:.1f}h")

    run_phase(args.phase2_config, override_load_path=phase1_ckpt)
    print(f"[curriculum] Phase 2 complete; total {(time.time()-t0)/3600:.1f}h")
    print("[curriculum] M3b-v6 curriculum done.")


if __name__ == "__main__":
    main()
