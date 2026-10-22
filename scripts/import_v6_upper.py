#!/usr/bin/env python3
"""Import only the rigid v6 upper assembly; retain K2's established leg tree.

Usage: python scripts/import_v6_upper.py /path/to/robot_v6_description
Then run phase1_mass.py and phase2_make_mjcf.py.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('export', type=Path)
    args = parser.parse_args()
    source = ET.parse(args.export / 'Robot_v6.urdf').getroot()
    old = ET.parse(ROOT / 'urdf/K2_fusion_export.urdf').getroot()
    # Corrected export: X=-Fusion Z, Y=-Fusion X, Z=Fusion Y.
    # This proper rotation (det=+1) maps back to the legacy Fusion frame.
    rotation = np.array([[0., -1., 0.], [0., 0., 1.], [-1., 0., 0.]])
    def hip_midpoint(robot):
        return np.mean([np.fromstring(robot.find(f"joint[@name='base_link_hip_pitch_{side}_joint']/origin").get('xyz'), sep=' ') for side in ('right', 'left')], axis=0)
    translation = hip_midpoint(old) - rotation @ hip_midpoint(source)
    parts = [('base_link', np.eye(4))]
    for joint in source.findall('joint'):
        if joint.get('type') != 'fixed' or joint.find('parent').get('link') != 'base_link':
            continue
        name = joint.find('child').get('link')
        # Existing leg mass accounting already includes both hip servos.
        origin = joint.find('origin')
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
        transform[:3, 3] = np.fromstring(origin.get('xyz'), sep=' ')
        parts.append((name, transform))
    align = np.eye(4)
    align[:3, :3], align[:3, 3] = rotation, translation
    cad_total = sum(float(source.find(f"link[@name='{n}']/inertial/mass").get('value')) for n, _ in parts if n != 'battery' and not n.startswith('Hip_pitch_motor'))
    meshes, properties, visuals = [], [], []
    for name, transform in parts:
        link = source.find(f"link[@name='{name}']")
        transform = align @ transform
        mesh = trimesh.load(args.export / link.find('collision/geometry/mesh').get('filename'), force='mesh')
        mesh.apply_transform(transform)
        meshes.append(mesh)
        for index, visual in enumerate(link.findall('visual')):
            region = trimesh.load(args.export / visual.find('geometry/mesh').get('filename'), force='mesh')
            region.apply_transform(transform)
            filename = f"base_link_v6_{name}_{index}.obj"
            region.export(ROOT / 'meshes/K2' / filename)
            visuals.append(dict(file=filename, rgba=visual.find('material/color').get('rgba')))
        if name.startswith('Hip_pitch_motor'):
            continue
        inertial = link.find('inertial')
        cad_mass = float(inertial.find('mass').get('value'))
        mass = 0.150 if name == 'battery' else 0.225 * cad_mass / cad_total
        origin = inertial.find('origin')
        com = transform[:3, :3] @ np.fromstring(origin.get('xyz'), sep=' ') + transform[:3, 3]
        a = inertial.find('inertia').attrib
        tensor = np.array([[float(a['ixx']), float(a['ixy']), float(a['ixz'])], [float(a['ixy']), float(a['iyy']), float(a['iyz'])], [float(a['ixz']), float(a['iyz']), float(a['izz'])]])
        orient = transform[:3, :3] @ Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
        properties.append((mass, com, orient @ tensor @ orient.T * mass / cad_mass))
    mass = sum(p[0] for p in properties)
    com = sum(m * c for m, c, _ in properties) / mass
    tensor = sum(i + m * (np.dot(c-com, c-com) * np.eye(3) - np.outer(c-com, c-com)) for m, c, i in properties)
    trimesh.util.concatenate(meshes).export(ROOT / 'meshes/K2/base_link_v6.obj')
    imu = next(t for n, t in parts if n == 'imu')
    config = dict(source='Robot_v6 updated color export, edited v2; corrected orientation and lowered holders', visuals=visuals, mass_kg=mass, com=com.tolist(), inertia=tensor.tolist(), imu_pos=(rotation @ imu[:3, 3] + translation).tolist(), parts=[n for n, _ in parts], alignment_translation=translation.tolist())
    (ROOT / 'urdf/base_link_v6.json').write_text(json.dumps(config, indent=2) + '\n')
    print(json.dumps(config, indent=2))


if __name__ == '__main__':
    main()
