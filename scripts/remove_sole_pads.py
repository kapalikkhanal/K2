"""Reverse scripts/sole_pads.py: strip TPU pads back out of the active MJCF.

`apply_mjcf` is forward-only -- deleting urdf/sole_pads.json just stops it from
running, leaving already-baked pads in the model. Run this first, with the pad
config still present, to restore the pad-free foot bodies:

    python scripts/remove_sole_pads.py

It undoes all five coupled edits: the pad contact geom, the disabled plastic
collision, the folded-in inertial, the lowered foot sites and the raised
k2_root. Every restored value comes from the config's own baseline fields, so
this is an exact inverse rather than re-derived numbers.
"""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from sole_pads import CONFIG, KEYS, fmt

ROOT = Path(__file__).resolve().parent.parent
MJCF = ROOT / "mjcf/k2_physics.xml"


def main():
    if not CONFIG.exists():
        raise SystemExit(f"Need {CONFIG} to invert exactly; restore it first.")
    config = json.loads(CONFIG.read_text())
    tree = ET.parse(MJCF)
    mj = tree.getroot()
    asset = mj.find("asset")
    removed = 0

    for side, pad in config["pads"].items():
        foot = mj.find(f".//body[@name='Feet_{side}']")
        mesh_name = f"foot_sole_{side}"
        contact_name = f"{side}_foot_collision"

        pad_geom = foot.find(f"geom[@mesh='{mesh_name}']")
        if pad_geom is None:
            print(f"{side}: no pad geom found, skipping")
            continue
        foot.remove(pad_geom)
        removed += 1

        # Re-enable the plastic sole under its original name and attributes.
        plastic = foot.find(f"geom[@name='{side}_foot_plastic_collision']")
        if plastic is not None:
            plastic.attrib.clear()
            plastic.attrib.update(pad["contact"])

        # Restore the pad-free inertial recorded before folding.
        baseline = pad["mjcf_inertial"]
        tensor = baseline["inertia"]
        inertial = foot.find("inertial")
        inertial.attrib.clear()
        inertial.set("mass", str(baseline["mass"]))
        inertial.set("pos", fmt(baseline["com"]))
        inertial.set("fullinertia", fmt([tensor[i][j] for _, i, j in KEYS]))

        # Lift the foot sites back out of the pad.
        for site_name, position in pad["sites"].items():
            foot.find(f"site[@name='{site_name}']").set("pos", fmt(position))

        stale = asset.find(f"mesh[@name='{mesh_name}']")
        if stale is not None:
            asset.remove(stale)

    mj.find(".//body[@name='k2_root']").set("pos", fmt(config["root_pos"]))
    ET.indent(tree, space="  ")
    tree.write(MJCF, encoding="utf-8")
    print(f"Removed {removed} sole pad(s); k2_root back to {fmt(config['root_pos'])}")


if __name__ == "__main__":
    main()
