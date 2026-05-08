"""Visual sanity check for the MuJoCo XML using the interactive viewer.

Run:
    python scripts/view_model.py

Use the spacebar to pause/unpause and drag with the mouse to interact.
"""

from pathlib import Path

import mujoco
import mujoco.viewer

XML = Path(__file__).resolve().parents[1] / "sim" / "models" / "triple_pendulum.xml"


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    # Start near the bottom equilibrium so gravity doesn't yeet the pendulum.
    data.qpos[1] = 3.14159
    mujoco.mj_forward(model, data)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
