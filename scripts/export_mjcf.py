"""Write the scene MJCF to assets/ssl_field.xml (or --out) for reading or for MuJoCo's viewer.

    python scripts/export_mjcf.py
    python -m mujoco.viewer --mjcf=assets/ssl_field.xml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mujoco_ssl.scene import build_scene_xml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("assets/ssl_field.xml"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_scene_xml())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
