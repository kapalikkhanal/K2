"""Apply imported TPU pads without changing the established foot bodies/joints."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / 'urdf/sole_pads.json'
KEYS = (("ixx",0,0),("iyy",1,1),("izz",2,2),("ixy",0,1),("ixz",0,2),("iyz",1,2))


def fmt(values):
    return ' '.join(f'{v:.10g}' for v in values)


def apply_urdf(robot):
    if not CONFIG.exists():
        return
    config = json.loads(CONFIG.read_text())
    for side, pad in config['pads'].items():
        name = f'foot_sole_{side}'
        for tag in ('link', 'joint'):
            existing = robot.find(f"{tag}[@name='{name if tag == 'link' else name + '_fixed'}']")
            if existing is not None:
                robot.remove(existing)
        link = ET.SubElement(robot, 'link', name=name)
        inertial = ET.SubElement(link, 'inertial')
        ET.SubElement(inertial, 'origin', xyz=fmt(pad['com']), rpy='0 0 0')
        ET.SubElement(inertial, 'mass', value=str(pad['mass_kg']))
        ET.SubElement(inertial, 'inertia', **{k:str(pad['inertia'][i][j]) for k,i,j in KEYS})
        for tag in ('visual', 'collision'):
            geom = ET.SubElement(link, tag)
            geometry = ET.SubElement(geom, 'geometry')
            ET.SubElement(geometry, 'mesh', filename='package://K2/meshes/K2/'+pad['file'])
            if tag == 'visual':
                material = ET.SubElement(geom, 'material', name='TPU')
                ET.SubElement(material, 'color', rgba=pad['rgba'])
        joint = ET.SubElement(robot, 'joint', name=name+'_fixed', type='fixed')
        ET.SubElement(joint, 'parent', link=f'Feet_{side}')
        ET.SubElement(joint, 'child', link=name)
        ET.SubElement(joint, 'origin', xyz='0 0 0', rpy='0 0 0')


def apply_mjcf(mj):
    if not CONFIG.exists():
        return
    config = json.loads(CONFIG.read_text())
    asset = mj.find('asset')
    for side, pad in config['pads'].items():
        foot = mj.find(f".//body[@name='Feet_{side}']")
        name = f'foot_sole_{side}'
        if asset.find(f"mesh[@name='{name}']") is None:
            ET.SubElement(asset, 'mesh', name=name, file=pad['file'])
        contact_name = f'{side}_foot_collision'
        contact = foot.find(f"geom[@name='{contact_name}']")
        if contact.get('mesh') != name:
            contact.set('name', f'{side}_foot_plastic_collision')
            contact.set('contype', '0')
            contact.set('conaffinity', '0')
            contact = ET.SubElement(foot, 'geom', **pad['contact'])
            contact.set('name', contact_name)
            contact.set('mesh', name)
            contact.attrib.pop('pos', None)
            contact.attrib.pop('quat', None)
            contact.set('rgba', pad['rgba'])
        # Fold the fixed pad into Feet so body-based contact sensors still work.
        baseline = pad['mjcf_inertial']
        m0, m1 = baseline['mass'], pad['mass_kg']
        c0, c1 = np.array(baseline['com']), np.array(pad['com'])
        com = (m0*c0 + m1*c1)/(m0+m1)
        tensor = np.array(baseline['inertia']) + np.array(pad['inertia'])
        for mass, center in ((m0,c0),(m1,c1)):
            delta = center-com
            tensor += mass*(np.dot(delta,delta)*np.eye(3)-np.outer(delta,delta))
        inertia = foot.find('inertial')
        inertia.attrib.clear()
        inertia.set('mass', str(m0+m1))
        inertia.set('pos', fmt(com))
        inertia.set('fullinertia', fmt([tensor[i,j] for _,i,j in KEYS]))
        for site_name, position in pad['sites'].items():
            xyz = np.array(position)
            xyz[1] -= pad['thickness_m']
            foot.find(f"site[@name='{site_name}']").set('pos', fmt(xyz))
    root = mj.find(".//body[@name='k2_root']")
    pos = np.array(config['root_pos'])
    pos[2] += max(p['thickness_m'] for p in config['pads'].values())
    root.set('pos', fmt(pos))
