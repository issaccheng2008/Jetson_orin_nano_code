"""相机参数加载 — config/cameras.json（多摄像头配置）。

用法:
    from camera_config import load as load_camera
    cam = load_camera()                 # 取 active 配置
    cam = load_camera("laptop_test")    # 指定配置
    # 或用环境变量 CAMERA_PROFILE 切换

配置缺失时回退默认值（USB 实车相机参数）。
"""
import json
import os

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "config", "cameras.json")

_DEFAULTS = {
    "index": 0,
    "width": 1280,
    "height": 720,
    "vfov_deg": 55.876,
    "mount_height_cm": 32.5,
    "pitch_deg": 45.0,
    "distance_calib": {"a": 1.0, "b": 0.0},
}


def load(profile=None):
    """返回当前摄像头参数字典（含 profile 名）。"""
    cam = dict(_DEFAULTS)
    cam["profile"] = "default"
    try:
        with open(_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
        key = profile or os.environ.get("CAMERA_PROFILE") or cfg.get("active")
        entry = dict(cfg["cameras"][key])
        cam.update({k: v for k, v in entry.items() if v is not None})
        cam["profile"] = key
    except Exception:
        pass
    return cam


_calib_cache = {}


def distance_calib(profile=None):
    """距离线性校正系数 (a, b)。"""
    key = profile or os.environ.get("CAMERA_PROFILE")
    if key not in _calib_cache:
        c = load(profile).get("distance_calib") or {}
        _calib_cache[key] = (float(c.get("a", 1.0)), float(c.get("b", 0.0)))
    return _calib_cache[key]


def to_true_z(z_est, profile=None):
    """相机模型算出的距离 → 地面真值 cm。

    模型（cy=图高/2、fy 由 vfov 反推的合成针孔）系统性低估距离，
    20cm 处 −1.8%、75cm 处 −9.1%。系数由地面尺子实测拟合，见
    scripts/fit_camera_distance.py。
    """
    a, b = distance_calib(profile)
    return a * z_est + b


def to_model_z(z_true, profile=None):
    """地面真值 → 相机模型读数 cm。用于把"希望它在多少厘米处"反推成像素门槛。"""
    a, b = distance_calib(profile)
    return (z_true - b) / a
