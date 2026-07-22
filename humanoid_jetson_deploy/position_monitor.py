"""Live motor-position plot and CSV logging for the deployment loop."""

from __future__ import annotations

import csv
from datetime import datetime
import multiprocessing
from pathlib import Path
import queue
from typing import Sequence

import numpy as np


IMU_ACCEL_LABELS = ("accel x", "accel y", "accel z")
IMU_ORIENTATION_LABELS = ("roll", "pitch", "yaw")
IMU_ACCEL_FIELDS = tuple(f"imu_accel_{axis}_m_s2" for axis in "xyz")
IMU_ORIENTATION_FIELDS = tuple(f"imu_{axis}_rad" for axis in IMU_ORIENTATION_LABELS)
IMU_ACCEL_TOGGLE = "IMU acceleration"
IMU_ORIENTATION_TOGGLE = "IMU orientation"


class PositionCsvLogger:
    """Write motor positions and requested IMU measurements to one CSV per run."""

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
            + list(IMU_ACCEL_FIELDS)
            + list(IMU_ORIENTATION_FIELDS)
        )

    def write(
        self,
        elapsed_s: float,
        step: int,
        state_sequence: int,
        target_motor_rad: np.ndarray,
        actual_motor_rad: np.ndarray,
        imu_accel_m_s2: np.ndarray,
        imu_orientation_rpy_rad: np.ndarray,
    ) -> None:
        target = np.asarray(target_motor_rad, dtype=np.float32).reshape(len(self.joint_names))
        actual = np.asarray(actual_motor_rad, dtype=np.float32).reshape(len(self.joint_names))
        acceleration = np.asarray(imu_accel_m_s2, dtype=np.float32).reshape(3)
        orientation = np.asarray(imu_orientation_rpy_rad, dtype=np.float32).reshape(3)
        self._writer.writerow(
            [
                datetime.now().astimezone().isoformat(timespec="milliseconds"),
                f"{elapsed_s:.6f}",
                int(step),
                int(state_sequence),
                *[f"{value:.8f}" for value in target],
                *[f"{value:.8f}" for value in actual],
                *[f"{value:.8f}" for value in acceleration],
                *[f"{value:.8f}" for value in orientation],
            ]
        )

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()


def load_position_log(
    path: str | Path,
    joint_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load elapsed time, motor positions, acceleration, and orientation."""
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
        available_fields = set(reader.fieldnames or ())
        imu_fields = (*IMU_ACCEL_FIELDS, *IMU_ORIENTATION_FIELDS)
        present_imu_fields = [field for field in imu_fields if field in available_fields]
        if present_imu_fields and len(present_imu_fields) != len(imu_fields):
            missing_imu_fields = [field for field in imu_fields if field not in available_fields]
            raise ValueError(
                f"{log_path} is missing IMU columns: {', '.join(missing_imu_fields)}"
            )
        has_imu = len(present_imu_fields) == len(imu_fields)
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
        if has_imu:
            acceleration = np.array(
                [[float(row[field]) for field in IMU_ACCEL_FIELDS] for row in rows],
                dtype=np.float32,
            )
            orientation = np.array(
                [[float(row[field]) for field in IMU_ORIENTATION_FIELDS] for row in rows],
                dtype=np.float32,
            )
        else:
            acceleration = np.empty((len(rows), 0), dtype=np.float32)
            orientation = np.empty((len(rows), 0), dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{log_path} contains invalid numeric data: {exc}") from exc

    arrays = (elapsed_s, targets, actuals, acceleration, orientation)
    if not all(np.all(np.isfinite(values)) for values in arrays):
        raise ValueError(f"{log_path} contains non-finite motor-position or IMU data")
    return arrays


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
        self._acceleration: list[np.ndarray] = []
        self._orientation: list[np.ndarray] = []
        self._closed = False

        self.figure = plt.figure(figsize=(14, 9))
        if getattr(self.figure.canvas, "required_interactive_framework", None) is None:
            backend = plt.get_backend()
            plt.close(self.figure)
            raise RuntimeError(f"Matplotlib backend {backend!r} cannot open a window")
        try:
            self.figure.canvas.manager.set_window_title("Humanoid motor and IMU monitor")
        except AttributeError:
            pass
        self.axes = self.figure.add_axes((0.07, 0.55, 0.61, 0.39))
        self.accel_axes = self.figure.add_axes((0.07, 0.30, 0.61, 0.18), sharex=self.axes)
        self.orientation_axes = self.figure.add_axes(
            (0.07, 0.07, 0.61, 0.18), sharex=self.axes
        )
        selector_axes = self.figure.add_axes((0.72, 0.07, 0.27, 0.87))
        selector_axes.set_title("Visible data")

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

        imu_colors = ("tab:red", "tab:green", "tab:blue")
        self._accel_lines = [
            self.accel_axes.plot([], [], color=color, label=label)[0]
            for label, color in zip(IMU_ACCEL_LABELS, imu_colors)
        ]
        self._orientation_lines = [
            self.orientation_axes.plot([], [], color=color, label=label)[0]
            for label, color in zip(IMU_ORIENTATION_LABELS, imu_colors)
        ]

        selector_labels = self.joint_names + (IMU_ACCEL_TOGGLE, IMU_ORIENTATION_TOGGLE)
        self._buttons = CheckButtons(
            selector_axes,
            selector_labels,
            default_selected + (True, True),
        )
        self._buttons.on_clicked(self._toggle_data)
        self.axes.set_title("STM32 motor positions: commanded target vs measured actual")
        self.axes.set_ylabel("Motor position (rad)")
        self.accel_axes.set_title("IMU acceleration in policy frame")
        self.accel_axes.set_ylabel("Acceleration (m/s²)")
        self.orientation_axes.set_title("IMU fused orientation in policy frame")
        self.orientation_axes.set_ylabel("Angle (rad)")
        self.orientation_axes.set_xlabel("Time since policy start (s)")
        for axis in (self.axes, self.accel_axes, self.orientation_axes):
            axis.grid(True, alpha=0.3)
        self.accel_axes.legend(loc="upper left", ncol=3)
        self.orientation_axes.legend(loc="upper left", ncol=3)
        self._refresh_legend()
        self.figure.canvas.mpl_connect("close_event", self._on_close)
        plt.show(block=False)
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def _on_close(self, _event) -> None:
        self._closed = True

    def _toggle_data(self, label: str) -> None:
        if label == IMU_ACCEL_TOGGLE:
            visible = not self.accel_axes.get_visible()
            self.accel_axes.set_visible(visible)
        elif label == IMU_ORIENTATION_TOGGLE:
            visible = not self.orientation_axes.get_visible()
            self.orientation_axes.set_visible(visible)
        else:
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
        imu_accel_m_s2: np.ndarray,
        imu_orientation_rpy_rad: np.ndarray,
    ) -> None:
        if self._closed:
            return

        target = np.asarray(target_motor_rad, dtype=np.float32).reshape(len(self.joint_names))
        actual = np.asarray(actual_motor_rad, dtype=np.float32).reshape(len(self.joint_names))
        acceleration = np.asarray(imu_accel_m_s2, dtype=np.float32).reshape(3)
        orientation = np.asarray(imu_orientation_rpy_rad, dtype=np.float32).reshape(3)
        self._times.append(float(elapsed_s))
        self._targets.append(target.copy())
        self._actuals.append(actual.copy())
        self._acceleration.append(acceleration.copy())
        self._orientation.append(orientation.copy())

        cutoff = elapsed_s - self.history_seconds
        first_kept = 0
        while first_kept < len(self._times) and self._times[first_kept] < cutoff:
            first_kept += 1
        if first_kept:
            del self._times[:first_kept]
            del self._targets[:first_kept]
            del self._actuals[:first_kept]
            del self._acceleration[:first_kept]
            del self._orientation[:first_kept]

        targets = np.asarray(self._targets)
        actuals = np.asarray(self._actuals)
        acceleration_values = np.asarray(self._acceleration)
        orientation_values = np.asarray(self._orientation)
        for index, (target_line, actual_line) in enumerate(
            zip(self._target_lines, self._actual_lines)
        ):
            target_line.set_data(self._times, targets[:, index])
            actual_line.set_data(self._times, actuals[:, index])
        for index, line in enumerate(self._accel_lines):
            line.set_data(self._times, acceleration_values[:, index])
        for index, line in enumerate(self._orientation_lines):
            line.set_data(self._times, orientation_values[:, index])

        right = max(self.history_seconds, elapsed_s)
        self.orientation_axes.set_xlim(max(0.0, right - self.history_seconds), right)
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
        self._set_limits(self.accel_axes, acceleration_values, minimum_padding=0.1)
        self._set_limits(self.orientation_axes, orientation_values, minimum_padding=0.05)

        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    @staticmethod
    def _set_limits(axis, values: np.ndarray, minimum_padding: float) -> None:
        low = float(np.min(values))
        high = float(np.max(values))
        padding = max(minimum_padding, 0.1 * max(high - low, 0.01))
        axis.set_ylim(low - padding, high + padding)

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
        imu_accel_m_s2: np.ndarray,
        imu_orientation_rpy_rad: np.ndarray,
    ) -> None:
        if not self._process.is_alive():
            return
        sample = (
            float(elapsed_s),
            np.asarray(target_motor_rad, dtype=np.float32).copy(),
            np.asarray(actual_motor_rad, dtype=np.float32).copy(),
            np.asarray(imu_accel_m_s2, dtype=np.float32).copy(),
            np.asarray(imu_orientation_rpy_rad, dtype=np.float32).copy(),
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
