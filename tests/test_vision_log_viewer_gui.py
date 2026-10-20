"""Exercise actual Tk widgets, keys, slider and asynchronous folder loading."""
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

try:
    import tkinter as tk
except ImportError:
    tk = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'new_vision/scripts'))
if tk is not None:
    from view_vision_log import VisionLogViewer


@unittest.skipUnless(tk is not None, 'Tkinter is not installed on this machine')
class VisionLogGuiTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f'Tk display unavailable: {exc}')
        self.app = VisionLogViewer(self.root)
        self.addCleanup(self.app.close)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.callback_errors = []
        self.root.report_callback_exception = lambda *args: self.callback_errors.append(args)

    def write_log(self, run='run_a', count=31):
        path = self.folder / 'vision' / run / 'line_frames.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(json.dumps(dict(frame=i + 100, process_monotonic_s=i * .1,
            host_time_ns=1790000000000000000 + i * 100000000, wz=.1,
            measurement=dict(measurement_valid=True, left_seen=True, right_seen=True,
                steering_heading_deg=5., steering_filtered_heading_deg=3.)))
            for i in range(count)), encoding='utf-8')
        return path

    def wait_import(self):
        deadline = time.monotonic() + 5
        while self.app._busy and time.monotonic() < deadline:
            self.root.update()
            time.sleep(.01)
        self.root.update()
        self.assertFalse(self.app._busy, 'background import timed out')
        self.assertEqual(self.callback_errors, [])

    def test_folder_keys_scale_row_selection_and_separate_runs(self):
        self.write_log()
        other = self.write_log('run_b', count=3)
        self.app.import_path(self.folder)
        self.wait_import()
        self.assertEqual(len(self.app.table.get_children()), 20)
        self.assertEqual(len(self.app.table['columns']), 8)
        self.assertEqual(self.app.table.selection(), ('0',))
        self.root.focus_force()
        self.root.update()
        self.root.event_generate('<KeyPress-f>')
        self.root.update()
        self.assertEqual(self.app.cursor.index, 5)
        self.root.event_generate('<KeyPress-F>')
        self.root.update()
        self.assertEqual(self.app.cursor.index, 10)
        self.root.event_generate('<KeyPress-d>')
        self.root.update()
        self.assertEqual(self.app.cursor.index, 5)
        self.root.event_generate('<KeyPress-D>')
        self.root.update()
        self.assertEqual(self.app.cursor.index, 0)
        for widget in (self.app.table, self.app.scale, self.app.run_choice, self.app.open_button):
            widget.focus_force()
            self.root.update()
            widget.event_generate('<KeyPress-f>')
            self.root.update()
            self.assertEqual(self.app.cursor.index, 5)
            widget.event_generate('<KeyPress-d>')
            self.root.update()
            self.assertEqual(self.app.cursor.index, 0)
        self.app.scale.set(1.6)
        self.root.update()
        self.assertEqual(self.app.cursor.index, 16)
        self.assertEqual(self.app.table.selection(), ('16',))
        self.assertEqual(self.app.table.get_children()[0], '6')
        self.app.forward_button.invoke()
        self.assertEqual(self.app.cursor.index, 21)
        self.app.step(100)
        self.assertEqual(self.app.cursor.index, 30)
        self.assertTrue(self.app.forward_button.instate(['disabled']))
        self.app.run_choice.current(1)
        self.app.run_choice.event_generate('<<ComboboxSelected>>')
        self.wait_import()
        self.assertEqual(self.app.log.path, other.resolve())
        self.assertEqual(len(self.app.table.get_children()), 3)
        self.assertEqual(self.app.cursor.index, 0)

    def test_failed_import_keeps_previous_log_and_reenables_import(self):
        self.write_log(count=1)
        self.app.import_path(self.folder)
        self.wait_import()
        previous = self.app.log
        self.assertTrue(self.app.scale.instate(['disabled']))
        self.assertTrue(self.app.back_button.instate(['disabled']))
        self.assertTrue(self.app.forward_button.instate(['disabled']))
        with patch('view_vision_log.messagebox.showerror') as error:
            self.app.import_path(self.folder / 'absent')
            self.wait_import()
            error.assert_called_once()
        self.assertIs(self.app.log, previous)
        self.assertTrue(self.app.open_button.instate(['!disabled']))
        self.assertEqual(len(self.app.table.get_children()), 1)

    def test_rowhdg_and_actual_robot_command_are_displayed_separately(self):
        path = self.write_log(count=1)
        row = json.loads(path.read_text(encoding='utf-8'))
        row['measurement'].update(rowhdg_deg=0., rowhdg_valid=False, steering_heading_deg=-32.1)
        row.update(robot_cmd_status='written', robot_cmd_wz=-.17)
        path.write_text(json.dumps(row), encoding='utf-8')
        self.app.import_path(self.folder)
        self.wait_import()
        self.assertEqual(self.app.table.set('0', 'rowhdg'), '+0.000 (无效)')
        self.assertEqual(self.app.table.set('0', 'heading'), '-32.100')
        self.assertEqual(self.app.table.set('0', 'robot_wz'), '-0.170')

    def test_raw_demand_column_shows_logged_value_or_missing(self):
        path = self.write_log(count=2)
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        rows[0]['measurement'].update(steering_demand_deg=9.327595389097295,
                                      steering_filtered_demand_deg=13.55)
        path.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
        self.app.import_path(self.folder)
        self.wait_import()
        self.assertEqual(self.app.table.set('0', 'demand'), '+9.328')
        self.assertEqual(self.app.table.set('1', 'demand'), '未记录')


if __name__ == '__main__':
    unittest.main()
