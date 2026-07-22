#!/usr/bin/env python3
"""Open a saved motor-position CSV in an interactive target/actual plot."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


DEPLOY_DIR = Path(__file__).resolve().parents[1]
if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))

import config  # noqa: E402
from position_monitor import (  # noqa: E402
    IMU_ACCEL_LABELS,
    IMU_ACCEL_TOGGLE,
    IMU_ORIENTATION_LABELS,
    IMU_ORIENTATION_TOGGLE,
    load_position_log,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_file",
        nargs="?",
        help="CSV to open; omit to use the newest log in --log-dir",
    )
    parser.add_argument(
        "--log-dir",
        default="logs/motor_positions",
        help="Directory searched when log_file is omitted",
    )
    return parser.parse_args()


def resolve_log_path(log_file: str | None, log_dir: str) -> Path:
    if log_file:
        path = Path(log_file).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Motor-position log not found: {path}")
        return path

    directory = Path(log_dir).expanduser()
    candidates = list(directory.glob("motor_positions_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No motor-position CSV logs found in {directory}. "
            "Pass a CSV path explicitly if it is stored elsewhere."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def main() -> int:
    args = parse_args()
    try:
        log_path = resolve_log_path(args.log_file, args.log_dir)
        elapsed_s, targets, actuals, acceleration, orientation = load_position_log(
            log_path, config.JOINT_NAMES
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    has_imu = acceleration.shape == (len(elapsed_s), 3) and orientation.shape == (
        len(elapsed_s),
        3,
    )

    import matplotlib.pyplot as plt
    from matplotlib.widgets import CheckButtons

    figure = plt.figure(figsize=(14, 9) if has_imu else (14, 7))
    try:
        figure.canvas.manager.set_window_title(f"Motor and IMU log: {log_path.name}")
    except AttributeError:
        pass
    if has_imu:
        axes = figure.add_axes((0.07, 0.55, 0.61, 0.39))
        accel_axes = figure.add_axes((0.07, 0.30, 0.61, 0.18), sharex=axes)
        orientation_axes = figure.add_axes((0.07, 0.07, 0.61, 0.18), sharex=axes)
        selector_axes = figure.add_axes((0.72, 0.07, 0.27, 0.87))
    else:
        axes = figure.add_axes((0.07, 0.12, 0.61, 0.82))
        accel_axes = figure.add_axes((0.07, 0.30, 0.61, 0.18), sharex=axes)
        orientation_axes = figure.add_axes((0.07, 0.07, 0.61, 0.18), sharex=axes)
        selector_axes = figure.add_axes((0.72, 0.10, 0.27, 0.84))
    selector_axes.set_title("Visible data")

    colors = plt.get_cmap("tab20").colors
    target_lines = []
    actual_lines = []
    default_selected = tuple("knee" in name.lower() for name in config.JOINT_NAMES)
    for index, name in enumerate(config.JOINT_NAMES):
        color = colors[index % len(colors)]
        target_line, = axes.plot(
            elapsed_s,
            targets[:, index],
            color=color,
            linewidth=1.8,
            label=f"{name} target",
        )
        actual_line, = axes.plot(
            elapsed_s,
            actuals[:, index],
            color=color,
            linewidth=1.4,
            linestyle="--",
            label=f"{name} actual",
        )
        target_line.set_visible(default_selected[index])
        actual_line.set_visible(default_selected[index])
        target_lines.append(target_line)
        actual_lines.append(actual_line)

    imu_colors = ("tab:red", "tab:green", "tab:blue")
    accel_lines = (
        [
            accel_axes.plot(
                elapsed_s, acceleration[:, index], color=color, label=label
            )[0]
            for index, (label, color) in enumerate(zip(IMU_ACCEL_LABELS, imu_colors))
        ]
        if has_imu
        else []
    )
    orientation_lines = (
        [
            orientation_axes.plot(
                elapsed_s, orientation[:, index], color=color, label=label
            )[0]
            for index, (label, color) in enumerate(
                zip(IMU_ORIENTATION_LABELS, imu_colors)
            )
        ]
        if has_imu
        else []
    )

    def refresh_legend_and_limits() -> None:
        visible_lines = [
            line
            for pair in zip(target_lines, actual_lines)
            for line in pair
            if line.get_visible()
        ]
        old_legend = axes.get_legend()
        if old_legend is not None:
            old_legend.remove()
        if visible_lines:
            axes.legend(
                visible_lines,
                [line.get_label() for line in visible_lines],
                loc="upper left",
                ncol=2,
            )
            values = np.concatenate([line.get_ydata() for line in visible_lines])
            low = float(np.min(values))
            high = float(np.max(values))
            padding = max(0.05, 0.1 * max(high - low, 0.01))
            axes.set_ylim(low - padding, high + padding)

    def toggle_data(label: str) -> None:
        if label == IMU_ACCEL_TOGGLE:
            accel_axes.set_visible(not accel_axes.get_visible())
        elif label == IMU_ORIENTATION_TOGGLE:
            orientation_axes.set_visible(not orientation_axes.get_visible())
        else:
            index = config.JOINT_NAMES.index(label)
            visible = not target_lines[index].get_visible()
            target_lines[index].set_visible(visible)
            actual_lines[index].set_visible(visible)
            refresh_legend_and_limits()
        figure.canvas.draw_idle()

    selector_labels = tuple(config.JOINT_NAMES)
    selector_states = default_selected
    if has_imu:
        selector_labels += (IMU_ACCEL_TOGGLE, IMU_ORIENTATION_TOGGLE)
        selector_states += (True, True)
    buttons = CheckButtons(
        selector_axes,
        selector_labels,
        selector_states,
    )
    buttons.on_clicked(toggle_data)
    axes.set_title(f"Saved STM32 motor positions — {log_path.name}")
    axes.set_ylabel("Motor position (rad)")
    if not has_imu:
        axes.set_xlabel("Time since policy start (s)")
    accel_axes.set_title("IMU acceleration in policy frame")
    accel_axes.set_ylabel("Acceleration (m/s²)")
    if has_imu:
        accel_axes.legend(
            accel_lines,
            [line.get_label() for line in accel_lines],
            loc="upper left",
            ncol=3,
        )
    orientation_axes.set_title("IMU fused orientation in policy frame")
    orientation_axes.set_ylabel("Angle (rad)")
    orientation_axes.set_xlabel("Time since policy start (s)")
    if has_imu:
        orientation_axes.legend(
            orientation_lines,
            [line.get_label() for line in orientation_lines],
            loc="upper left",
            ncol=3,
        )
    else:
        accel_axes.set_visible(False)
        orientation_axes.set_visible(False)
    for axis in (axes, accel_axes, orientation_axes):
        axis.grid(True, alpha=0.3)
    axes.set_xlim(float(elapsed_s[0]), float(elapsed_s[-1]))
    refresh_legend_and_limits()

    print(f"Loaded motor-position log: {log_path}")
    print(f"Samples: {len(elapsed_s)}, duration: {elapsed_s[-1] - elapsed_s[0]:.3f} s")
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
