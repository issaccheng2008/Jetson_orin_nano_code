#!/usr/bin/env python3
"""Offline vision log GUI: python view_vision_log.py [session_folder]."""
from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, font, messagebox, ttk

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jetson'))
from vision_log_data import FrameCursor, discover_logs, load_log


def number(value):
    return '未记录' if value is None else f'{value:+.3f}'


def clock_text(frame):
    if frame.host_time_ns is None:
        return f'+{frame.elapsed_s:.3f} s'
    try:
        stamp = datetime.fromtimestamp(frame.host_time_ns / 1e9).astimezone()
        return stamp.strftime('%H:%M:%S.') + f'{frame.host_time_ns // 1_000_000 % 1000:03d}'
    except (OSError, ValueError, OverflowError):
        return f'+{frame.elapsed_s:.3f} s'


class VisionLogViewer:
    def __init__(self, root):
        self.root = root
        self.log = self.cursor = None
        self.paths = ()
        self.folder = None
        self.results = queue.Queue()
        self._updating_scale = False
        self._busy = False
        self._missing_execution = False
        self.root.title('Vision 日志查看器')
        width = min(1340, max(1000, root.winfo_screenwidth() - 80))
        height = min(880, max(650, root.winfo_screenheight() - 90))
        root.geometry(f'{width}x{height}')
        root.minsize(1000, 650)
        root.configure(background='#f1f4f8')
        self._style()
        self._build()
        root.bind('<KeyPress>', self._key_press)
        self._poll_id = root.after(80, self._poll_results)
        root.protocol('WM_DELETE_WINDOW', self.close)

    def _style(self):
        families = set(font.families(self.root))
        self.family = next((name for name in ('Microsoft YaHei UI', 'Noto Sans CJK SC',
                           'WenQuanYi Micro Hei') if name in families), 'TkDefaultFont')
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure('.', font=(self.family, 10), background='#f1f4f8', foreground='#25364a')
        style.configure('Title.TLabel', font=(self.family, 20, 'bold'), foreground='#183f64')
        style.configure('Muted.TLabel', foreground='#607286')
        style.configure('Current.TLabel', font=(self.family, 12, 'bold'), foreground='#176b92')
        style.configure('TButton', padding=(14, 7))
        style.configure('Accent.TButton', background='#176b92', foreground='white')
        style.map('Accent.TButton', background=[('active', '#125677'), ('disabled', '#cbd6df')])
        style.configure('Treeview', rowheight=26, background='white', fieldbackground='white',
                        font=(self.family, 10), borderwidth=0)
        style.configure('Treeview.Heading', background='#e2eaf2', foreground='#29445d',
                        font=(self.family, 10, 'bold'), padding=(6, 9))
        style.map('Treeview', background=[('selected', '#c6e7f5')],
                  foreground=[('selected', '#0c3b56')])

    def _build(self):
        shell = ttk.Frame(self.root, padding=(22, 16))
        shell.pack(fill='both', expand=True)
        header = ttk.Frame(shell)
        header.pack(fill='x')
        ttk.Label(header, text='Vision 日志查看器', style='Title.TLabel').pack(side='left')
        self.open_button = ttk.Button(header, text='导入时间文件夹', style='Accent.TButton',
                                      command=self.choose_folder)
        self.open_button.pack(side='right')
        self.file_button = ttk.Button(header, text='打开 JSONL', command=self.choose_file)
        self.file_button.pack(side='right', padx=10)
        self.source_text = tk.StringVar(value='导入一次运行的时间文件夹，即可查看其中的 vision 日志。')
        ttk.Label(shell, textvariable=self.source_text, style='Muted.TLabel',
                  wraplength=1180).pack(fill='x', pady=(8, 12))

        run_bar = ttk.Frame(shell)
        run_bar.pack(fill='x', pady=(0, 10))
        ttk.Label(run_bar, text='运行记录').pack(side='left', padx=(0, 10))
        self.run_choice = ttk.Combobox(run_bar, state='disabled', width=55)
        self.run_choice.pack(side='left', fill='x', expand=True)
        self.run_choice.bind('<<ComboboxSelected>>', self._choose_run)
        self.count_text = tk.StringVar(value='尚未导入')
        ttk.Label(run_bar, textvariable=self.count_text, style='Muted.TLabel').pack(side='right', padx=(16, 0))

        self.current_text = tk.StringVar(value='选择时间点后，这里显示当前帧')
        ttk.Label(shell, textvariable=self.current_text, style='Current.TLabel').pack(anchor='w', pady=(0, 8))
        table_box = ttk.Frame(shell)
        table_box.pack(fill='both', expand=True)
        columns = ('time', 'rowhdg', 'heading', 'filtered', 'demand', 'line', 'vision_wz', 'robot_wz')
        self.table = ttk.Treeview(table_box, columns=columns, show='tree headings', height=20,
                                  selectmode='browse')
        self.table.heading('#0', text='帧')
        self.table.column('#0', width=72, minwidth=60, stretch=False, anchor='center')
        for name, label, width in zip(columns,
                ('时间', 'rowhdg 原始角 (°)', '偏转角 (°)', '滤波后偏转角 (°)', '查表目标角 (°)', '识别线的情况',
                 'vision 输出角速度 (rad/s)', '实机采用角速度 (rad/s)'),
                (174, 170, 120, 150, 150, 140, 205, 205)):
            self.table.heading(name, text=label)
            self.table.column(name, width=width, minwidth=width, anchor='center')
        vertical = ttk.Scrollbar(table_box, orient='vertical', command=self.table.yview)
        horizontal = ttk.Scrollbar(table_box, orient='horizontal', command=self.table.xview)
        self.table.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.table.grid(row=0, column=0, sticky='nsew')
        vertical.grid(row=0, column=1, sticky='ns')
        horizontal.grid(row=1, column=0, sticky='ew')
        table_box.rowconfigure(0, weight=1)
        table_box.columnconfigure(0, weight=1)
        self.table.tag_configure('even', background='#f5f8fb')
        self.table.tag_configure('lost', foreground='#b13f3f')
        self.table.tag_configure('selected', background='#c6e7f5', foreground='#0c3b56')
        self.table.bind('<ButtonRelease-1>', self._click_row)

        timeline_bar = ttk.Frame(shell)
        timeline_bar.pack(fill='x', pady=(14, 4))
        self.back_button = ttk.Button(timeline_bar, text='←  后退 5 帧 [D]',
                                      command=lambda: self.step(-5), state='disabled')
        self.back_button.pack(side='left')
        self.forward_button = ttk.Button(timeline_bar, text='前进 5 帧 [F]  →',
                                         command=lambda: self.step(5), state='disabled')
        self.forward_button.pack(side='right')
        self.position_text = tk.StringVar(value='时间轴 · 拖动选择时间')
        ttk.Label(timeline_bar, textvariable=self.position_text).pack(expand=True)
        self.scale = ttk.Scale(shell, orient='horizontal', from_=0, to=1,
                               command=self._scrub, state='disabled')
        self.scale.pack(fill='x', pady=3)
        ends = ttk.Frame(shell)
        ends.pack(fill='x')
        self.start_text, self.end_text = tk.StringVar(value='起点'), tk.StringVar(value='终点')
        ttk.Label(ends, textvariable=self.start_text, style='Muted.TLabel').pack(side='left')
        ttk.Label(ends, textvariable=self.end_text, style='Muted.TLabel').pack(side='right')
        self.status_text = tk.StringVar(value='支持 LOG="vision"；角度为日志原值，实机角速度为采用的指令值。')
        ttk.Label(shell, textvariable=self.status_text, wraplength=1180,
                  style='Muted.TLabel').pack(fill='x', pady=(10, 0))

    def choose_folder(self):
        path = filedialog.askdirectory(parent=self.root, title='选择当次日志的时间文件夹',
                                        initialdir=str(self.folder) if self.folder else None)
        if path:
            self.import_path(path)

    def choose_file(self):
        path = filedialog.askopenfilename(parent=self.root, title='打开 vision 逐帧日志',
            filetypes=[('Vision 日志', '*.jsonl')],
            initialdir=str(self.folder) if self.folder else None)
        if path:
            self.import_path(path)

    def _set_busy(self, busy):
        self._busy = busy
        for widget in (self.open_button, self.file_button):
            widget.configure(state='disabled' if busy else 'normal')
        self.run_choice.configure(state='readonly' if self.paths and not busy else 'disabled')

    def import_path(self, path, *, discover=True):
        if self._busy:
            return
        self._set_busy(True)
        self.status_text.set('正在读取日志…')

        def read():
            try:
                paths = discover_logs(path) if discover else self.paths
                selected = paths[0] if discover else Path(path)
                log = load_log(selected)
                self.results.put(('loaded', (paths, log)))
            except Exception as exc:
                self.results.put(('error', str(exc)))

        threading.Thread(target=read, name='vision-log-import', daemon=True).start()

    def _poll_results(self):
        try:
            kind, payload = self.results.get_nowait()
        except queue.Empty:
            pass
        else:
            self._set_busy(False)
            if kind == 'loaded':
                self.paths, self.log = payload
                self.folder = self.log.path.parent
                # Shortest common parent gives readable labels while keeping runs distinct.
                common = Path(os.path.commonpath([str(path.parent) for path in self.paths]))
                if len(self.paths) == 1:
                    common = common.parent
                self.run_choice.configure(values=[str(path.relative_to(common)) for path in self.paths],
                                          state='readonly')
                self.run_choice.current(self.paths.index(self.log.path))
                self.cursor = FrameCursor(self.log.frames)
                self._missing_execution = all(frame.robot_status == '未记录' for frame in self.log.frames)
                self.source_text.set(str(self.log.path))
                self.count_text.set(f'{len(self.log.frames):,} 帧 · {len(self.paths)} 段运行')
                duration = self.log.frames[-1].elapsed_s
                self.scale.configure(to=max(duration, .001), state='normal' if duration > 0 else 'disabled')
                self.start_text.set(f'{clock_text(self.log.frames[0])}  /  +0.000 s')
                self.end_text.set(f'{clock_text(self.log.frames[-1])}  /  +{duration:.3f} s')
                self.refresh()
                self.root.focus_set()
            else:
                if self.log is not None:
                    self.run_choice.current(self.paths.index(self.log.path))
                    self.refresh()
                else:
                    self.status_text.set('导入失败，请重新选择日志文件夹。')
                messagebox.showerror('无法打开日志', payload, parent=self.root)
        self._poll_id = self.root.after(80, self._poll_results)

    def _choose_run(self, _event=None):
        index = self.run_choice.current()
        if index >= 0:
            self.import_path(self.paths[index], discover=False)

    def _key_press(self, event):
        if event.state & 0xC:  # Leave Ctrl/Alt combinations to the platform.
            return None
        key = event.keysym.lower()
        if key in ('d', 'f'):
            self.step(-5 if key == 'd' else 5)
            return 'break'
        return None

    def step(self, delta):
        if self.cursor is not None:
            self.cursor.step(delta)
            self.refresh()

    def _scrub(self, value):
        if self.cursor is not None and not self._updating_scale:
            self.cursor.seek_time(float(value))
            # Keep the thumb under the pointer while dragging; keys/row clicks snap it.
            self.refresh(sync_scale=False)

    def _click_row(self, event):
        row = self.table.identify_row(event.y)
        if row and self.cursor is not None:
            self.cursor.select(int(row))
            self.refresh()
            self.table.focus_set()

    def refresh(self, *, sync_scale=True):
        if self.cursor is None:
            return
        cursor = self.cursor
        selected = cursor.frames[cursor.index]
        start, end = cursor.window()
        self.table.delete(*self.table.get_children())
        for index in range(start, end):
            frame = cursor.frames[index]
            robot = number(frame.robot_wz) if frame.robot_wz is not None else frame.robot_status
            rowhdg = number(frame.rowhdg_deg)
            if frame.rowhdg_deg is not None and frame.rowhdg_valid is False:
                rowhdg += ' (无效)'
            tags = ['even'] if index % 2 == 0 else []
            if frame.line_state == '丢线':
                tags.append('lost')
            if index == cursor.index:
                tags.append('selected')
            self.table.insert('', 'end', iid=str(index), text=str(frame.frame), tags=tags,
                values=(clock_text(frame), rowhdg, number(frame.heading_deg), number(frame.filtered_heading_deg),
                        number(frame.demand_deg), frame.line_state, number(frame.vision_wz), robot))
        self.table.selection_set(str(cursor.index))
        self.table.see(str(cursor.index))
        if sync_scale:
            self._updating_scale = True
            try:
                self.scale.set(selected.elapsed_s)
            finally:
                self._updating_scale = False
        self.back_button.configure(state='normal' if cursor.index > 0 else 'disabled')
        self.forward_button.configure(state='normal' if cursor.index < len(cursor.frames) - 1 else 'disabled')
        date = ''
        if selected.host_time_ns is not None:
            try:
                date = datetime.fromtimestamp(selected.host_time_ns / 1e9).astimezone().strftime('%Y-%m-%d %z  ·  ')
            except (OSError, ValueError, OverflowError):
                pass
        self.current_text.set(f'{date}当前帧 #{selected.frame}  ·  {clock_text(selected)}  ·  '
                              f'显示第 {start + 1}–{end} 条（共 {end - start} 帧）')
        self.position_text.set(f'+{selected.elapsed_s:.3f} s  /  {cursor.frames[-1].elapsed_s:.3f} s'
                               f'     ·     {cursor.index + 1:,} / {len(cursor.frames):,}')
        notes = ['rowhdg 为 Close+Fit 逐行拟合原始角；“无效”数值仍保留，不代表有效方向。'
                 '查表目标角优先显示实际 steering_table_input_deg（启用 F/C 加权时为加权角）；'
                 '旧日志缺少该字段时显示原始 steering_demand_deg；滤波后偏转角仍表示线路方向，并非 F。'
                 '旧日志仍按原有控制逻辑生成命令。实机列为采用的指令角速度。']
        if self._missing_execution:
            notes.append('此旧日志未记录实机反馈，实机列显示“未记录”。')
        if self.log.skipped_rows:
            notes.append(f'已跳过 {self.log.skipped_rows} 行无效/不完整数据。')
        if self.log.clock_adjustments:
            notes.append(f'{self.log.clock_adjustments} 处时间倒退已固定到上一帧时间。')
        if not self._busy:
            self.status_text.set(' '.join(notes))

    def close(self):
        self.root.after_cancel(self._poll_id)
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log_path', nargs='?', help='当次时间文件夹、vision/run 目录或 line_frames.jsonl')
    args = parser.parse_args()
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise SystemExit(f'无法启动桌面窗口：{exc}。请在带桌面的电脑运行（Linux 需安装 python3-tk）。') from exc
    app = VisionLogViewer(root)
    if args.log_path:
        root.after(100, lambda: app.import_path(args.log_path))
    root.mainloop()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
