"""Standalone debug perception/steering defaults, independent of robot wiring."""
from copy import deepcopy
from contextlib import contextmanager
import os
from pathlib import Path

# Explicit scope: transport, policy launch, startup sequence, recording and
# hold-still are operational controls and must never be overwritten here.
DEBUG_FIELDS = tuple('''
camera_async camera camera_device card_camera card_camera_device
width height camera_height_cm camera_pitch_deg camera_vfov_deg
camera_exposure_mode camera_exposure_ms camera_brightness camera_contrast
camera_saturation camera_sharpness camera_white_balance_mode camera_white_balance_k camera_power_line_hz
vx max_wz steer_full_scale_cm yaw_sign step_len_cm preview_gain lost_hold_s deriv_pole
bias_cm bias_gate_px bias_dead_px bias_straight_cm max_lateral_cm
line_pitch line_pitch_hold_s no_shape_detect no_red_detect
card_hold_ms card_stop_ms card_trigger_frac card_trigger_dist_cm card_tilt_ms card_settle_ms
card_cold_start stop_cool_down card_failure_cooldown_s card_every_stopped
shape_proposal_mode card_clear_calls shape_every center_dead_cm max_wz_right single_line_gain
wz_mode heading_lookahead_cm heading_far_cm heading_near_cm heading_regions_cm steering_angle_wz_table
segment_regions_cm heading_right_tolerance_deg heading_left_tolerance_deg heading_corridor_cm
heading_full_scale_deg heading_left_wz heading_right_wz heading_straight_wz
position_recovery_cm position_recovery_full_scale_cm position_confirm_frames
wz_fire_cm wz_stop_cm wz_step wz_turn_s wz_gap_s wz_allow_right anticipation_clip
line_preprocess lane_observation_mode lane_strictness_pct lane_gap_fill lane_visible_direction_fallback
lane_gap_max_cm straight_road_guard straight_near_cm straight_far_cm straight_confirm_s straight_exit_s
straight_hold_s initial_boundary_side single_boundary_tracking line_adaptive_c shape_preprocess photometric_mode
photometric_lut_dir lane_fit lane_fit_segments lane_fit_near_cm lane_fit_far_cm
steering_filter_mode steering_filter_algorithm steering_filter_robust_tau_s steering_filter_robust_window_s
steering_filter_robust_slew_deg_s steering_filter_min_hz steering_filter_max_hz steering_filter_beta
steering_filter_derivative_hz steering_filter_position_tau_s steering_hysteresis_deg steering_enter_deg
steering_exit_deg steering_loss_mode steering_loss_max_s steering_loss_history_s steering_loss_fallback_wz
steering_segment_fallback steering_command_window_s steering_command_median command_min_hold_s
'''.split())

# Freeze environment-sensitive settings to the measured standalone debug run.
# All other values come from the very same argparse defaults as that invocation.
DEBUG_ENV_DEFAULTS = dict(camera=0, camera_device='/dev/robot_main_camera',
    card_camera=-1, card_camera_device=None, width=1280, height=720,
    camera_height_cm=32.5, camera_pitch_deg=45., camera_vfov_deg=55.876,
    step_len_cm=8., preview_gain=0., deriv_pole=.78, bias_cm=3.,
    bias_gate_px=12., bias_dead_px=6., bias_straight_cm=1., max_lateral_cm=0.,
    single_line_gain=1., heading_far_cm=29., heading_near_cm=None,
    heading_regions_cm='', steering_angle_wz_table='',
    segment_regions_cm='[[20,32],[32,44],[44,56]]', anticipation_clip=.5,
    line_preprocess='legacy', shape_preprocess='selective', photometric_mode='lut',
    line_pitch_hold_s=2.5, card_hold_ms=3000., card_stop_ms=3000., card_trigger_dist_cm=43.,
    card_tilt_ms=1800., card_settle_ms=100., stop_cool_down=5., card_failure_cooldown_s=.5,
    card_every_stopped=2, card_clear_calls=4, shape_every=2, center_dead_cm=4.,
    photometric_lut_dir=str(Path(__file__).resolve().parents[1] / 'config/photometric_mapping'))


def apply_debug_profile(args, parser):
    if args.vision_profile != 'debug':
        return
    previous = {name: parser.get_default(name) for name in DEBUG_ENV_DEFAULTS}
    parser.set_defaults(**DEBUG_ENV_DEFAULTS)
    try:
        defaults = parser.parse_args([])
    finally:
        parser.set_defaults(**previous)
    for name in DEBUG_FIELDS:
        setattr(args, name, deepcopy(getattr(defaults, name)))
    # Receive feedback for actions, readiness and robot_cmd logs, but match
    # --attitude-port 0's static geometry instead of feeding pitch into vision.
    args.attitude_pitch_correction = False


@contextmanager
def runtime_debug_environment(args):
    """Pin legacy detector environment consumers for the duration of this run."""
    if args.vision_profile != 'debug':
        yield
        return
    settings = dict(CAMERA_PROFILE='usb_main', SHAPE_MAX_DIST_CM='80.0',
                    PHOTOMETRIC_LUT_DIR=args.photometric_lut_dir)
    previous = {key: os.environ.get(key) for key in settings}
    original_camera = shape_module = None
    try:
        os.environ.update(settings)
        # Shape geometry caches the camera at module import. Refresh it also
        # when an embedded caller has already imported the detector previously.
        import shape_detector as shape_module
        from camera_config import load
        original_camera = shape_module._CAM
        shape_module._CAM = load('usb_main')
        yield
    finally:
        if shape_module is not None and original_camera is not None:
            shape_module._CAM = original_camera
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
