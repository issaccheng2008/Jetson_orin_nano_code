"""Low-latency USB-camera opening for NVIDIA Jetson and normal Linux hosts."""

import os
import platform

import cv2


def _is_jetson():
    return platform.machine().lower() in ("aarch64", "arm64") and (
        os.path.exists("/etc/nv_tegra_release")
        or os.path.exists("/etc/nvidia-container-runtime/host-files-for-container.d")
    )


def opencv_has_gstreamer():
    for line in cv2.getBuildInformation().splitlines():
        if "GStreamer:" in line:
            return "YES" in line.upper()
    return False


def mjpeg_gstreamer_pipeline(device, width, height, fps):
    """Decode UVC MJPEG with Jetson's nvv4l2decoder, then expose BGR."""
    return (
        f"v4l2src device={device} io-mode=2 ! "
        f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
        "jpegparse ! nvv4l2decoder mjpeg=true ! "
        "nvvidconv ! video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def _open_v4l2(index, width, height, fps):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def open_camera(index, width, height, fps, backend="auto"):
    """Open a camera and return ``(capture, selected_backend)``.

    ``auto`` tries Jetson's hardware MJPEG decode first and safely falls back
    to regular V4L2.  Use ``CAMERA_BACKEND=v4l2`` if a camera's MJPEG stream is
    not accepted by the NVIDIA decoder.
    """
    backend = str(backend).strip().lower()
    if backend not in ("auto", "gstreamer", "v4l2"):
        raise ValueError("CAMERA_BACKEND must be auto, gstreamer, or v4l2")

    try_gstreamer = backend == "gstreamer" or (
        backend == "auto" and _is_jetson() and opencv_has_gstreamer())
    if try_gstreamer:
        device = f"/dev/video{index}"
        pipeline = mjpeg_gstreamer_pipeline(device, width, height, fps)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            return cap, "gstreamer-nvv4l2decoder"
        cap.release()
        if backend == "gstreamer":
            raise RuntimeError(
                "GStreamer camera pipeline failed. Check gst-inspect-1.0 "
                "nvv4l2decoder and the camera's MJPEG mode with v4l2-ctl.")
        print("[camera] accelerated GStreamer open failed; falling back to V4L2")

    cap = _open_v4l2(index, width, height, fps)
    return cap, "v4l2"
