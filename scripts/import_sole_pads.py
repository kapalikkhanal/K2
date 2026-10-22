#!/usr/bin/env python3
"""Import v6 TPU soles, registering each to the retained foot collision mesh.

Mass defaults to a provisional 10 g per pad; override with --mass-grams.
Thickness defaults to the CAD thickness; override with --thickness-mm.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from sole_pads import ROOT, CONFIG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('export', type=Path)
    parser.add_argument('--mass-grams', type=float, default=10)
    parser.add_argument('--thickness-mm', type=float)
    args = parser.parse_args()
    if args.mass_grams <= 0 or (args.thickness_mm is not None and args.thickness_mm <= 0):
        parser.error('mass and thickness must be positive')
    source = ET.parse(args.export/'Robot_v6.urdf').getroot()
    target = ET.parse(ROOT/'urdf/K2_phase1.urdf').getroot()
    mj = ET.parse(ROOT/'mjcf/k2_physics.xml').getroot()
    # Keep original inertials/sites for deterministic, repeatable regeneration.
    config = json.loads(CONFIG.read_text()) if CONFIG.exists() else dict(root_pos=np.fromstring(mj.find(".//body[@name='k2_root']").get('pos'), sep=' ').tolist(), pads={})
    for side in ('left', 'right'):
        name = f'foot_sole_{side}'
        link = source.find(f"link[@name='{name}']")
        parent = source.find(f"link[@name='Feet_{side}']")
        rotation = np.array([[0.,-1,0],[0,0,1],[-1,0,0]])
        if side == 'left':
            rotation = np.diag([-1,1,-1]) @ rotation
        new_foot = trimesh.load(args.export/parent.find('collision/geometry/mesh').get('filename'), force='mesh')
        new_foot.vertices = new_foot.vertices @ rotation.T
        old_geom = target.find(f"link[@name='Feet_{side}']/collision")
        old_mesh = old_geom.find('geometry/mesh')
        old_foot = trimesh.load(ROOT/'meshes/K2'/Path(old_mesh.get('filename')).name, force='mesh')
        old_foot.vertices = old_foot.vertices*np.fromstring(old_mesh.get('scale','1 1 1'), sep=' ') + np.fromstring(old_geom.find('origin').get('xyz'), sep=' ')
        if not np.allclose(new_foot.extents, old_foot.extents, atol=1e-5):
            raise ValueError(f'{side} foot geometry changed; registration requires review')
        offset = old_foot.bounds.mean(axis=0)-new_foot.bounds.mean(axis=0)
        joint = source.find(f"joint[@name='{name}_fixed']/origin")
        pad = trimesh.load(args.export/link.find('collision/geometry/mesh').get('filename'), force='mesh')
        pad.vertices = (pad.vertices+np.fromstring(joint.get('xyz'),sep=' '))@rotation.T + offset
        thickness = pad.extents[1]
        if args.thickness_mm is not None:
            thickness = args.thickness_mm/1000
            top = pad.bounds[1,1]
            pad.vertices[:,1] = top+(pad.vertices[:,1]-top)*thickness/pad.extents[1]
        # Remove tessellation-scale offsets: pad top exactly meets the plastic sole.
        pad.vertices[:,1] += old_foot.bounds[0,1]-pad.bounds[1,1]
        mass = args.mass_grams/1000
        inertia = pad.moment_inertia*(mass/pad.mass)
        if pad.volume <= 0 or np.linalg.eigvalsh(inertia).min() <= 0:
            raise ValueError('Sole must be a closed, outward-wound mesh')
        filename = name+'_tpu.stl'
        pad.export(ROOT/'meshes/K2'/filename)
        old = config['pads'].get(side)
        body = mj.find(f".//body[@name='Feet_{side}']")
        if old is None:
            inertial = body.find('inertial')
            quat = np.fromstring(inertial.get('quat', '1 0 0 0'), sep=' ')
            orient = Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
            tensor = orient@np.diag(np.fromstring(inertial.get('diaginertia'),sep=' '))@orient.T
            baseline = dict(mass=float(inertial.get('mass')), com=np.fromstring(inertial.get('pos'),sep=' ').tolist(), inertia=tensor.tolist())
            sites = {s.get('name'):np.fromstring(s.get('pos'),sep=' ').tolist() for s in body.findall('site')}
            contact = dict(body.find(f"geom[@name='{side}_foot_collision']").attrib)
        else:
            baseline, sites, contact = old['mjcf_inertial'], old['sites'], old['contact']
        config['pads'][side] = dict(file=filename, mass_kg=mass, thickness_m=thickness, com=pad.center_mass.tolist(), inertia=inertia.tolist(), rgba=link.find('visual/material/color').get('rgba'), mjcf_inertial=baseline, sites=sites, contact=contact)
    config['mass_note'] = 'Provisional mass; replace with measured pad weight using --mass-grams.'
    config['source'] = 'Robot_v6 updated color export, foot_sole_left/right'
    CONFIG.write_text(json.dumps(config, indent=2)+'\n')
    print(f'Imported two TPU pads at {args.mass_grams:g} g each.')


if __name__ == '__main__':
    main()
