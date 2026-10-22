#!/usr/bin/env python3
"""Archive the exact firmware, source, configuration and policy for a trial.

Run after building and before uploading:
  python esp32-s3/tools/archive_trial.py --out trial-build.zip
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    esp = ROOT/'esp32-s3'
    match = re.search(r'^// source: (.+)$', (esp/'src/policy_weights.h').read_text(), re.M)
    policy = ROOT / match.group(1) if match else ROOT/'policies/last_policy.onnx'
    if not policy.is_file(): parser.error(f'Missing exported policy source: {policy}')
    source = [*sorted((esp/'src').glob('*')), *sorted((esp/'tools').glob('*.py')),
              esp/'platformio.ini', esp/'partitions_robot.csv', esp/'docs/field_logging.md',
              policy, ROOT/'hardware/calibration.json',
              ROOT/'hardware/imu_level_calibration.json', ROOT/'hardware/joint_limits.json',
              ROOT/'mjcf/k2_physics.xml', ROOT/'robot_data.yaml']
    build = esp/'.pio/build/robot'
    binaries = [build/name for name in ('firmware.bin', 'firmware.elf', 'bootloader.bin', 'partitions.bin')]
    for path in binaries:
        if not path.is_file(): parser.error(f'Build robot first: missing {path}')
    # A stale binary must never be presented as the build of modified sources.
    compile_inputs = [*sorted((esp/"src").glob("*")), esp/"platformio.ini", esp/"partitions_robot.csv"]
    newest = max(p.stat().st_mtime for p in compile_inputs if p.is_file())
    if min(p.stat().st_mtime for p in binaries[:2]) < newest:
        parser.error('Firmware is older than source/configuration. Run pio run -d esp32-s3 -e robot first.')
    files = [p for p in source+binaries if p.is_file()]
    manifest = {'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'files': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                'note': 'Content hashes identify working-tree files, including uncommitted changes.'}
    with zipfile.ZipFile(args.out, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files: archive.write(path, str(path.relative_to(ROOT)))
        archive.writestr('manifest.json', json.dumps(manifest, indent=2))
    print(f'Archived {len(files)} files to {args.out}')


if __name__ == '__main__': main()
