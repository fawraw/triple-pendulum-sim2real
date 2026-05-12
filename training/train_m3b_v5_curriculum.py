"""M3b-v5: two-phase curriculum training for all 8 equilibria.

Phase 1 (1M steps): extreme oversample of EP4/EP6 (hard_ep_weight=50) to build
a strong prior for those configurations. All previous approaches failed to push
EP4/EP6 above 10% in random-mode — this forced-focus approach is the escalation.

Phase 2 (2M steps): fine-tune from Phase 1 checkpoint on all 8 EPs with moderate
oversample (hard_ep_weight=3) and lower LR to prevent catastrophic forgetting.

Run from project root:

    MUJOCO_GL=osmesa python -m training.train_m3b_v5_curriculum \\
        --phase1_config training/configs/m3b_v5_phase1.yaml \\
        --phase2_config training/configs/m3b_v5_phase2.yaml
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


def run_phase(cfg_path: str, load_model_path: str = "") -> str:
    """Run one training phase and return the path of the final checkpoint."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    if load_model_path:
        cfg["load_model_path"] = load_model_path
        # Write a temporary config with the resolved path so train_main can read it
        tmp_path = Path(cfg_path).with_suffix(".tmp.yaml")
        with open(tmp_path, "w") as f:
            yaml.dump(cfg, f)
        cfg_to_use = str(tmp_path)
    else:
        cfg_to_use = cfg_path

    print(f"\n{'='*60}")
    print(f"Starting: {cfg.get('stage', cfg_path)}")
    if load_model_path:
        print(f"  warm-start from: {load_model_path}")
    print(f"{'='*60}\n")

    # train_main writes the final.zip to checkpoints/<run_name>/final.zip
    # We capture the run_name by monkeypatching time.strftime, but the simpler
    # approach is to just look for the newest final.zip after training.
    before = set(Path(ROOT / "checkpoints").glob("*/final.zip"))
    train_main(cfg_to_use)
    after = set(Path(ROOT / "checkpoints").glob("*/final.zip"))

    if load_model_path:
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
    # --config is accepted as an alias for --phase1_config so the generic bootstrap
    # script (which always passes --config $TP_STAGE_CONFIG) works without changes.
    p.add_argument("--config", "--phase1_config", dest="phase1_config",
                   default="training/configs/m3b_v5_phase1.yaml")
    p.add_argument("--phase2_config", default="training/configs/m3b_v5_phase2.yaml")
    args = p.parse_args()

    t0 = time.time()

    print("[curriculum] === M3b-v5 two-phase curriculum ===")
    phase1_ckpt = run_phase(args.phase1_config)
    print(f"[curriculum] Phase 1 complete in {(time.time()-t0)/3600:.1f}h")

    run_phase(args.phase2_config, load_model_path=phase1_ckpt)
    print(f"[curriculum] Phase 2 complete in {(time.time()-t0)/3600:.1f}h total")
    print("[curriculum] M3b-v5 curriculum done.")


if __name__ == "__main__":
    main()
