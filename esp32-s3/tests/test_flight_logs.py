import importlib.util
import math
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

spec = importlib.util.spec_from_file_location('flight_logs', Path(__file__).parents[1]/'tools/flight_logs.py')
logs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(logs)

DEFAULTS = [-.149, 0., 0., -.611, .462, 0., -.153, 0., 0., -.611, .458, 0.]


def record(kind=2, seq=0, version=2, session=123, text='test event', values=None):
    h = logs.HEADER.pack(logs.MAGIC, version, kind, session, seq, 1000000+seq*40000,
                         0x49, 20000, 3000, 1000, 5000, 0xfff, 0, 0, 0, 0, 0)
    v = [0.0]*111
    v[5] = -1
    v[80] = .03
    v[83] = .1
    for index, x in (values or {}).items(): v[index] = x
    payload = struct.pack('<111f', *v) if kind != 1 else text.encode().ljust(444, b'\0')
    body = h + payload
    return body + struct.pack('<I', zlib.crc32(body))


def identity(session=123, version=3):
    # One joint per event record, exactly as logIdentity() writes them.
    return b''.join(record(1, seq=j, version=version, session=session,
                           text=f'joint index={j} id={5-j} sign=-1 home_raw=1698 '
                                f'default_rad={DEFAULTS[j]:.9g} min_rad=-1 max_rad=1 '
                                f'action_scale=0.2') for j in range(12))


class LogTests(unittest.TestCase):
    def test_position_only_capture_keeps_unavailable_fields_missing(self):
        values = {i: math.nan for i in range(86, 110)}
        values.update({6: math.nan, 18: math.nan, 3: math.nan})
        raw = identity() + record(seq=12, version=3, values=values)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'run.bin').write_bytes(raw)
            summary = logs.decode(root/'run.bin', root/'analysis')
            self.assertIsNone(summary['sessions']['123']['control_rate_servo']['0']['current_raw'])
            self.assertFalse((root/'analysis/servos.csv').exists())
        row = logs.columns(logs.read_records(raw)[0][-1], dict(enumerate(DEFAULTS)))
        for key in ('q_rad_0', 'qd_rad_s_0', 'current_raw_0', 'load_signed_raw_0', 'pitch_deg'):
            self.assertTrue(math.isnan(row[key]), key)

    def test_crc_torn_and_erased_records(self):
        corrupt = bytearray(record());corrupt[100] ^= 1
        result, invalid = logs.read_records(record()+bytes(corrupt)+b'\xff'*512+record()[:200])
        self.assertEqual(len(result), 1)
        self.assertEqual(len(invalid), 2)
        self.assertEqual(result[0]['valid_mask'], 4095)

    def test_unknown_version_is_rejected(self):
        _, invalid = logs.read_records(record(version=4))
        self.assertEqual(invalid[0]['reason'], 'magic/version/CRC')

    def test_decode_session_and_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);source=root/'run.bin'
            source.write_bytes(record(1)+record(seq=1)+record(seq=3)+record(3,4))
            summary=logs.decode(source,root/'analysis')
            self.assertEqual(summary['valid_records'],4)
            session=summary['sessions']['123']
            self.assertEqual(session['sequence_gaps'],1)
            self.assertEqual(session['moving_records'],2)
            self.assertAlmostEqual(session['modes']['forward']['tilt_deg']['max'],5.729578,places=5)
            self.assertTrue((root/'analysis/control.csv').exists())
            self.assertTrue((root/'analysis/servos.csv').exists())
            with self.assertRaises(FileExistsError): logs.decode(source,root/'analysis')

    def test_stale_values_excluded_from_summary(self):
        data=bytearray(record(3));struct.pack_into('<H',data,44,0)
        struct.pack_into('<I',data,508,zlib.crc32(data[:508]))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'run.bin';source.write_bytes(data)
            summary=logs.decode(source,root/'analysis')
            self.assertIsNone(summary['sessions']['123']['servo_telemetry']['0']['voltage_v'])

    def test_v2_still_decodes_q_from_its_own_slots(self):
        # An existing capture must keep decoding after the format change, and its
        # slots 86..109 must never be reported as load/current.
        control = logs.columns(logs.read_records(record(values={86: .25, 98: 1.5}))[0][0])
        self.assertAlmostEqual(control['q_rad_0'], .25)
        self.assertAlmostEqual(control['qd_rad_s_0'], 1.5)
        self.assertNotIn('load_signed_raw_0', control)

    def test_v3_reconstructs_q_and_reports_load_current(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);source=root/'run.bin'
            source.write_bytes(identity()+record(seq=12, version=3, values={
                6: .05, 18: 1.5, 62: .2, 86: -200., 98: 350.}))
            summary = logs.decode(source, root/'analysis')
            rows = list(__import__('csv').DictReader((root/'analysis/control.csv').open()))
            self.assertEqual(len(rows), 1)
            # obs[6] is q minus the default pose, so q comes back exactly.
            self.assertAlmostEqual(float(rows[0]['q_rad_0']), .05+DEFAULTS[0], places=6)
            self.assertAlmostEqual(float(rows[0]['qd_rad_s_0']), 1.5)
            self.assertAlmostEqual(float(rows[0]['load_signed_raw_0']), -200.)
            self.assertAlmostEqual(float(rows[0]['current_raw_0']), 350.)
            self.assertAlmostEqual(float(rows[0]['goal_minus_measured_deg_0']),
                                   math.degrees(.2-(.05+DEFAULTS[0])), places=4)
            session = summary['sessions']['123']
            self.assertEqual(session['record_versions'], [3])
            self.assertEqual(session['default_pose_source'], 'session boot events')
            self.assertAlmostEqual(session['control_rate_servo']['0']['load_signed_raw']['min'], -200.)

    def test_v3_without_boot_events_yields_nan_q_not_wrong_q(self):
        control = logs.columns(logs.read_records(record(version=3, values={6: .05}))[0][0], None)
        self.assertTrue(math.isnan(control['q_rad_0']))
        self.assertTrue(math.isnan(control['goal_minus_measured_deg_0']))

    def test_mixed_version_partition_shares_the_recorded_pose(self):
        # What the board actually holds: v2 boot events from earlier sessions
        # followed by v3 records from the reflashed firmware.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);source=root/'run.bin'
            source.write_bytes(identity(session=1, version=2)
                               + record(seq=12, version=3, session=2, values={6: .05}))
            summary = logs.decode(source, root/'analysis')
            self.assertEqual(summary['sessions']['2']['default_pose_source'],
                             'other sessions in this file')
            rows = list(__import__('csv').DictReader((root/'analysis/control.csv').open()))
            self.assertAlmostEqual(float(rows[0]['q_rad_0']), .05+DEFAULTS[0], places=6)


if __name__ == '__main__': unittest.main()
