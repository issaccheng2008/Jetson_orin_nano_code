"""Live motor-position plot and CSV logging for the deployment loop."""

from __future__ import annotations

import csv
from datetime import datetime
import multiprocessing
from pathlib import Path
import queue
from typing import Sequence

import numpy as np


class PositionCsvLogger:
    """Write all target and measured motor positions to one CSV per run."""

    def __init__(self, log_dir: str | Path, joint_names: Sequence[str]) -> None:
        self.joint_names = tuple(joint_names)
        directory = Path(log_dir).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        self.path = directory / f"motor_positions_{timestamp}.csv"
        self._file = self.path.open("w", encoding="utf-8", newline="", buffering=1)
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            ["host_time_iso", "elapsed_s", "step", "state_sequence"]
            + [f"target_{name}_rad" for name in self.joint_names]
            + [f"actual_{name}_rad" for name in self.joint_names]
        )

    def write(
        self,
        elapsed_s: float,
        step: int,
        state_sequence: int,
        target_motor_rad: np.ndarray,
        actual_motor_rad: np.ndarray,
    ) -> None:
        target = np.asarray(target_motor_rad, dtype=np.float32).reshape(len(self.joint_names))
        actual = np.asarray(actual_motor_rad, dtype=np.float32).reshape(len(self.joint_names))
        self._writer.writerow(
            [
                datetime.now().astimezone().isoformat(timespec="milliseconds"),
                f"{elapsed_s:.6f}",
                int(step),
                int(state_sequence),
                *[f"{value:.8f}" for value in target],
                *[f"{value:.8f}" for value in actual],
            ]
        )

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()


def load_position_log(
    path: str | Path,
    joint_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load elapsed time, target positions, and actual positions from a run log."""
    log_path = Path(path).expanduser()
    names = tuple(joint_names)
    target_fields = [f"target_{name}_rad" for name in names]
    actual_fields = [f"actual_{name}_rad" for name in names]
    required_fields = ["elapsed_s", *target_fields, *actual_fields]

    with log_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_fields = [field for field in required_fields if field not in (reader.fieldnames or ())]
        if missing_fields:
            raise ValueError(
                f"{log_path} is missing required columns: {', '.join(missing_fields)}"
            )
        rows = list(reader)

    if not rows:
        raise ValueError(f"{log_path} contains no motor-position samples")

    try:
        elapsed_s = np.array([float(row["elapsed_s"]) for row in rows], dtype=np.float64)
        targets = np.array(
            [[float(row[field]) for field in target_fields] for row in rows],
            dtype=np.float32,
        )
        actuals = np.array(
            [[float(row[field]) for field in actual_fields] for row in rows],
            dtype=np.float32,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{log_path} contains invalid numeric data: {exc}") from exc

    if not np.all(np.isfinite(elapsed_s)) or not np.all(np.isfinite(targets)) or not np.all(
        np.isfinite(actuals)
    ):
        raise ValueError(f"{log_path} contains non-finite motor-position data")
    return elapsed_s, targets, actuals


class _PlotWindow:
    """Matplotlib window owned by the dedicated plotting process."""

    def __init__(
        self,
        joint_names: Sequence[str],
        history_seconds: float = 10.0,
    ) -> None:
        if history_seconds <= 0.0:
            raise ValueError("history_seconds must be positive")

        # Import lazily so --no-plot can run on a headless Jetson without a GUI backend.
        import matplotlib.pyplot as plt
        from matplotlib.widgets import CheckButtons

        self._plt = plt
        self.joint_names = tuple(joint_names)
        self.history_seconds = float(history_seconds)
        self._times: list[float] = []
        self._targets: list[np.ndarray] = []
        self._actuals: list[np.ndarray] = []
        self._closed = False

        self.figure = plt.figure(figsize=(14, 7))
        if getattr(self.figure.canvas, "required_interactive_framework", None) is None:
            backend = plt.get_backend()
            plt.close(self.figure)
            raise RuntimeError(f"Matplotlib backend {backend!r} cannot open a window")
        try:
            self.figure.canvas.manager.set_window_title("Humanoid motor position monitor")
        except AttributeError:
            pass
        self.axes = self.figure.add_axes((0.07, 0.12, 0.61, 0.82))
        selector_axes = self.figure.add_axes((0.72, 0.10, 0.27, 0.84))
        selector_axes.set_title("Visible motors")

        colors = plt.get_cmap("tab20").colors
        self._target_lines = []
        self._actual_lines = []
        default_selected = tuple("knee" in name.lower() for name in self.joint_names)
        for index, name in enumerate(self.joint_names):
            color = colors[index % len(colors)]
            target_line, = self.axes.plot(
                [], [], color=color, linewidth=1.8, label=f"{name} target"
            )
            actual_line, = self.axes.plot(
                [], [], color=color, linewidth=1.4, linestyle="--", label=f"{name} actual"
            )
            target_line.set_visible(default_selected[index])
            actual_line.set_visible(default_selected[index])
            self._target_lines.append(target_line)
            self._actual_lines.append(actual_line)

        self._buttons = CheckButtons(selector_axes, self.joint_names, default_selected)
        self._buttons.on_clicked(self._toggle_joint)
        self.axes.set_title("STM32 motor positions: commanded target vs measured actual")
        self.axes.set_xlabel("Time since policy start (s)")
        self.axes.set_ylabel("Motor position (rad)")
        self.axes.grid(True, alpha=0.3)
        self._refresh_legend()
        self.figure.canvas.mpl_connect("close_event", self._on_close)
        plt.show(block=False)
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def _on_close(self, _event) -> None:
        self._closed = True

    def _toggle_joint(self, label: str) -> None:
        index = self.joint_names.index(label)
        visible = not self._target_lines[index].get_visible()
        self._target_lines[index].set_visible(visible)
        self._actual_lines[index].set_visible(visible)
        self._refresh_legend()
        self.figure.canvas.draw_idle()

    def _refresh_legend(self) -> None:
        lines = [
            line
            for pair in zip(self._target_lines, self._actual_lines)
            for line in pair
            if line.get_visible()
        ]
        old_legend = self.axes.get_legend()
        if old_legend is not None:
            old_legend.remove()
        if lines:
            self.axes.legend(lines, [line.get_label() for line in lines], loc="upper left", ncol=2)

    def update(
        self,
        elapsed_s: float,
        target_motor_rad: np.ndarray,
        actual_motor_rad: np.ndarray,
    ) -> None:
        if self._closed:
            return

        target = np.asarray(target_motor_rad, dtype=np.float32).reshape(len(self.joint_names))
        actual = np.asarray(actual_motor_rad, dtype=np.float32).reshape(len(self.joint_names))
        self._times.append(float(elapsed_s))
        self._targets.append(target.copy())
        self._actuals.append(actual.copy())

        cutoff = elapsed_s - self.history_seconds
        first_kept = 0
        while first_kept < len(self._times) and self._times[first_kept] < cutoff:
            first_kept += 1
        if first_kept:
            del self._times[:first_kept]
            del self._targets[:first_kept]
            del self._actuals[:first_kept]

        targets = np.asarray(self._targets)
        actuals = np.asarray(self._actuals)
        for index, (target_line, actual_line) in enumerate(
            zip(self._target_lines, self._actual_lines)
        ):
            target_line.set_data(self._times, targets[:, index])
            actual_line.set_data(self._times, actuals[:, index])

        right = max(self.history_seconds, elapsed_s)
        self.axes.set_xlim(max(0.0, right - self.history_seconds), right)
        visible_values = []
        for index, line in enumerate(self._target_lines):
            if line.get_visible():
                visible_values.extend((targets[:, index], actuals[:, index]))
        if visible_values:
            values = np.concatenate(visible_values)
            low = float(np.min(values))
            high = float(np.max(values))
            padding = max(0.05, 0.1 * max(high - low, 0.01))
            self.axes.set_ylim(low - padding, high + padding)

        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def close(self) -> None:
        if not self._closed:
            self._plt.close(self.figure)
            self._closed = True


def _plot_process_main(
    sample_queue,
    ready_connection,
    joint_names: tuple[str, ...],
    history_seconds: float,
) -> None:
    try:
        window = _PlotWindow(joint_names, history_seconds)
        ready_connection.send("")
    except Exception as exc:
        ready_connection.send(f"{type(exc).__name__}: {exc}")
        ready_connection.close()
        return
    ready_connection.close()

    try:
        while not window._closed:
            latest_sample = None
            try:
                message = sample_queue.get(timeout=0.02)
                received_sample = True
            except queue.Empty:
                received_sample = False
            if received_sample:
                if message is None:
                    break
                latest_sample = message
                while True:
                    try:
                        message = sample_queue.get_nowait()
                    except queue.Empty:
                        break
                    if message is None:
                        return
                    latest_sample = message
            if latest_sample is not None:
                window.update(*latest_sample)
            else:
                window.figure.canvas.flush_events()
    finally:
        window.close()


class LivePositionPlot:
    """Non-blocking controller for the live plot's dedicated process."""

    def __init__(
        self,
        joint_names: Sequence[str],
        history_seconds: float = 10.0,
    ) -> None:
        if history_seconds <= 0.0:
            raise ValueError("history_seconds must be positive")

        context = multiprocessing.get_context("spawn")
        self._queue = context.Queue(maxsize=8)
        ready_receiver, ready_sender = context.Pipe(duplex=False)
        self._process = context.Process(
            target=_plot_process_main,
            args=(self._queue, ready_sender, tuple(joint_names), float(history_seconds)),
            name="motor-position-plot",
            daemon=True,
        )
        self._process.start()
        ready_sender.close()
        if not ready_receiver.poll(15.0):
            self._process.terminate()
            self._process.join(timeout=1.0)
            raise RuntimeError("plot process did not initialize within 15 seconds")
        error = ready_receiver.recv()
        ready_receiver.close()
        if error:
            self._process.join(timeout=1.0)
            raise RuntimeError(error)

    def update(
        self,
        elapsed_s: float,
        target_motor_rad: np.ndarray,
        actual_motor_rad: np.ndarray,
    ) -> None:
        if not self._process.is_alive():
            return
        sample = (
            float(elapsed_s),
            np.asarray(target_motor_rad, dtype=np.float32).copy(),
            np.asarray(actual_motor_rad, dtype=np.float32).copy(),
        )
        try:
            self._queue.put_nowait(sample)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(sample)
            except queue.Full:
                pass

    def is_open(self) -> bool:
        """Return whether the plotting window process is still running."""
        return self._process.is_alive()

    def close(self) -> None:
        if self._process.is_alive():
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait(None)
                except (queue.Empty, queue.Full):
                    pass
            self._process.join(timeout=1.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)
        self._queue.close()
