"""Windows DirectShow driver controls for the standalone calibration tool.

Uses ctypes only. Capture uses OpenCV CAP_DSHOW; controls enumerate the same
CLSID_VideoInputDeviceCategory in the same index order. All COM calls stay on
the main thread. GetRange capability flags, Set and Get come from the driver;
OpenCV set() return values are never used to verify exposure or white balance.
"""
from __future__ import annotations

import ctypes as ct
import math
import sys
import uuid

from photometric_capture import CalibrationError, UNKNOWN, utc_now


class _GUID(ct.Structure):
    _fields_ = [("data1", ct.c_uint32), ("data2", ct.c_uint16),
                ("data3", ct.c_uint16), ("data4", ct.c_ubyte * 8)]

    @classmethod
    def parse(cls, value):
        return cls.from_buffer_copy(uuid.UUID(value).bytes_le)


_LONG = ct.c_int32
_PTR = ct.c_void_p
_REF = ct.POINTER(_LONG)
# These are Windows SDK enum flags, not OpenCV's backend-specific mode values.
_AUTO, _MANUAL = 1, 2
_CAMERA_IID = "c6e13370-30ac-11d0-a18c-00a0c9118956"
_PROCAMP_IID = "c6e13360-30ac-11d0-a18c-00a0c9118956"


def _call(pointer, slot, argtypes, *args):
    if not pointer:
        raise CalibrationError("DirectShow returned a null COM interface")
    table = ct.cast(pointer, ct.POINTER(ct.POINTER(_PTR))).contents
    method = ct.WINFUNCTYPE(_LONG, _PTR, *argtypes)(table[slot])
    return method(pointer, *args)


def _check(hr, operation):
    if hr < 0:
        raise CalibrationError(f"{operation}: HRESULT 0x{hr & 0xffffffff:08X}")


def _release(pointer):
    if pointer:
        _call(pointer, 2, ())


def _query(pointer, iid):
    result = _PTR()
    guid = _GUID.parse(iid)
    _check(_call(pointer, 0, (ct.POINTER(_GUID), ct.POINTER(_PTR)),
                 ct.byref(guid), ct.byref(result)), "QueryInterface " + iid)
    return result


class DirectShowController:
    """CameraControl/VideoProcAmp controls with readback and best-effort restore."""

    def __init__(self, index, allow_exposure_rounding=False):
        if sys.platform != "win32":
            raise CalibrationError("DirectShow controls require native Windows Python")
        if index < 0:
            raise CalibrationError("camera index must be non-negative")
        self.index = index
        self.device = f"DirectShow:{index}"
        self.allow_exposure_rounding = allow_exposure_rounding
        self.catalog, self.initial, self.initial_native = {}, {}, {}
        self._properties, self._interfaces = {}, []
        self._filter = None
        self._com_owned = False
        self._ole = ct.WinDLL("ole32")
        self._ole.CoInitializeEx.argtypes = [_PTR, ct.c_uint32]
        self._ole.CoInitializeEx.restype = _LONG
        self._ole.CoUninitialize.argtypes = []
        self._ole.CoUninitialize.restype = None
        self._ole.CoCreateInstance.argtypes = [ct.POINTER(_GUID), _PTR, ct.c_uint32,
                                              ct.POINTER(_GUID), ct.POINTER(_PTR)]
        self._ole.CoCreateInstance.restype = _LONG
        self._ole.CoTaskMemFree.argtypes = [_PTR]
        self._ole.CoTaskMemFree.restype = None
        hr = self._ole.CoInitializeEx(None, 2)  # COINIT_APARTMENTTHREADED
        # OpenCV may already have initialized this thread in another apartment.
        if (hr & 0xffffffff) != 0x80010106:  # RPC_E_CHANGED_MODE
            _check(hr, "CoInitializeEx")
            self._com_owned = True
        try:
            self._bind_camera()
        except BaseException:
            self.close()
            raise

    def _bind_camera(self):
        enumerator, devices, moniker = _PTR(), _PTR(), _PTR()
        clsid = _GUID.parse("62be5d10-60eb-11d0-bd3b-00a0c911ce86")
        iid = _GUID.parse("29840822-5b84-11d0-bd3b-00a0c911ce86")
        category = _GUID.parse("860bb310-5d01-11d0-bd3b-00a0c911ce86")
        try:
            _check(self._ole.CoCreateInstance(ct.byref(clsid), None, 1,
                                              ct.byref(iid), ct.byref(enumerator)),
                   "create DirectShow device enumerator")
            hr = _call(enumerator, 3, (ct.POINTER(_GUID), ct.POINTER(_PTR), ct.c_uint32),
                       ct.byref(category), ct.byref(devices), 0)
            _check(hr, "enumerate video devices")
            if hr != 0 or not devices:
                raise CalibrationError("Windows reports no DirectShow camera")
            for _ in range(self.index + 1):
                _release(moniker)
                moniker = _PTR()
                fetched = ct.c_uint32()
                hr = _call(devices, 3, (ct.c_uint32, ct.POINTER(_PTR), ct.POINTER(ct.c_uint32)),
                           1, ct.byref(moniker), ct.byref(fetched))
                _check(hr, "enumerate camera index")
                if hr != 0 or fetched.value != 1:
                    raise CalibrationError(f"Windows has no DirectShow camera index {self.index}")
            display_name = _PTR()
            try:
                hr = _call(moniker, 20, (_PTR, _PTR, ct.POINTER(_PTR)),
                           None, None, ct.byref(display_name))
                if hr == 0 and display_name:
                    self.device = ct.wstring_at(display_name)
            finally:
                if display_name:
                    self._ole.CoTaskMemFree(display_name)
            filter_iid = _GUID.parse("56a86895-0ad4-11ce-b03a-0020af0ba770")
            self._filter = _PTR()
            _check(_call(moniker, 8, (_PTR, _PTR, ct.POINTER(_GUID), ct.POINTER(_PTR)),
                         None, None, ct.byref(filter_iid), ct.byref(self._filter)),
                   "bind camera controls")
        finally:
            _release(moniker)
            _release(devices)
            _release(enumerator)

    def _get(self, name):
        interface, property_id = self._properties[name]
        value, flags = _LONG(), _LONG()
        _check(_call(interface, 5, (_LONG, _REF, _REF), property_id,
                     ct.byref(value), ct.byref(flags)), f"read {name}")
        return value.value, flags.value

    @staticmethod
    def _exposure_ms(native):
        try:
            value = math.ldexp(1000.0, native)
        except OverflowError as exc:
            raise CalibrationError("driver exposure is outside the DirectShow log2 range") from exc
        if not math.isfinite(value) or value <= 0:
            raise CalibrationError("invalid DirectShow exposure readback")
        return value

    def query_initial_state(self):
        definitions = [
            (_CAMERA_IID, [(4, "exposure_log2_seconds"), (0, "pan"), (1, "tilt"),
                           (2, "roll"), (3, "zoom"), (5, "iris"), (6, "focus")]),
            (_PROCAMP_IID, [(0, "brightness"), (1, "contrast"), (2, "hue"),
                            (3, "saturation"), (4, "sharpness"), (5, "gamma"),
                            (6, "color_enable"), (7, "white_balance_temperature"),
                            (8, "backlight_compensation"), (9, "gain")]),
        ]
        for iid, properties in definitions:
            try:
                interface = _query(self._filter, iid)
            except CalibrationError:
                continue
            self._interfaces.append(interface)
            for property_id, name in properties:
                fields = [_LONG() for _ in range(5)]
                hr = _call(interface, 3, (_LONG, _REF, _REF, _REF, _REF, _REF),
                           property_id, *(ct.byref(field) for field in fields))
                if hr != 0:
                    continue  # Unsupported properties are not fabricated.
                low, high, step, default, caps = [field.value for field in fields]
                self._properties[name] = (interface, property_id)
                self.catalog[name] = {"kind": "DirectShow property", "min": low,
                                      "max": high, "step": step, "default": default,
                                      "capability_flags": caps,
                                      "menu": {flag: label for flag, label in
                                               ((_AUTO, "Auto"), (_MANUAL, "Manual"))
                                               if caps & flag}}
                try:
                    self.initial_native[name] = self._get(name)
                except CalibrationError:
                    self.initial_native[name] = (UNKNOWN, UNKNOWN)
        for name in ("exposure_log2_seconds", "white_balance_temperature"):
            if name not in self.catalog or self.initial_native[name][0] == UNKNOWN:
                raise CalibrationError(f"driver cannot query/restore required control: {name}")
        exp_info = self.catalog["exposure_log2_seconds"]
        self.catalog["exposure_ms"] = {
            "kind": "derived from driver log2-seconds exposure", "unit": "ms",
            "min": self._exposure_ms(exp_info["min"]),
            "max": self._exposure_ms(exp_info["max"])}
        self.catalog["exposure_mode"] = {"kind": "driver flags", "menu": exp_info["menu"]}
        self.catalog["white_balance_mode"] = {
            "kind": "driver flags", "menu": self.catalog["white_balance_temperature"]["menu"]}
        self.initial = self._flatten(self.initial_native)
        return {"device": self.device, "camera_index": self.index, "backend": "DirectShow",
                "controls": {name: {**info, "actual": self.initial.get(name, UNKNOWN)}
                             for name, info in self.catalog.items()},
                "native_initial": {name: {"value": v, "flags": f}
                                   for name, (v, f) in self.initial_native.items()},
                "captured_at": utc_now()}

    def _flatten(self, native):
        values = {name: value for name, (value, _) in native.items()}
        value, flags = native.get("exposure_log2_seconds", (UNKNOWN, UNKNOWN))
        values["exposure_ms"] = self._exposure_ms(value) if value != UNKNOWN else UNKNOWN
        values["exposure_mode"] = flags
        values["white_balance_mode"] = native.get("white_balance_temperature", (UNKNOWN, UNKNOWN))[1]
        return values

    def read_all(self):
        native = {}
        for name in self._properties:
            try:
                native[name] = self._get(name)
            except CalibrationError:
                native[name] = (UNKNOWN, UNKNOWN)
        return self._flatten(native)

    def _find(self, aliases):
        translations = {"auto_exposure": "exposure_mode",
                        "exposure_time_absolute": "exposure_ms",
                        "white_balance_automatic": "white_balance_mode"}
        return next((translations.get(name, name) for name in aliases
                     if translations.get(name, name) in self.catalog), None)

    def _require_mode(self, name, flag):
        if name not in self.catalog or not self.catalog[name]["capability_flags"] & flag:
            raise CalibrationError(f"{name}: driver does not support requested mode {flag}")

    def resolve_fixed_exposure(self, requested_ms):
        if not math.isfinite(requested_ms) or requested_ms <= 0:
            raise CalibrationError("fixed exposure must be finite and positive")
        self._require_mode("exposure_log2_seconds", _MANUAL)
        info = self.catalog["exposure_log2_seconds"]
        low, high, step = info["min"], info["max"], info["step"]
        if high < low or step <= 0:
            raise CalibrationError("driver exposure range/step is invalid")
        log_value = math.log2(requested_ms / 1000.0)
        position = (log_value - low) / step
        last = (high - low) // step
        candidates = {low + min(last, max(0, n)) * step
                      for n in (math.floor(position), math.ceil(position))}
        native = min(candidates, key=lambda n: abs(self._exposure_ms(n) - requested_ms))
        actual_ms = self._exposure_ms(native)
        rounded = not math.isclose(actual_ms, requested_ms, rel_tol=1e-9, abs_tol=1e-9)
        if rounded and not self.allow_exposure_rounding:
            raise CalibrationError(
                f"Windows DirectShow cannot represent {requested_ms:g} ms; nearest supported "
                f"exposure is {actual_ms:g} ms (driver value {native}). "
                "Use that --fixed-exposure-ms value, or explicitly add --allow-exposure-rounding.")
        return {"requested_ms": requested_ms, "selected_ms": actual_ms,
                "native_value": native, "rounded": rounded,
                "unit": "log2 seconds", "rounding_allowed": self.allow_exposure_rounding}

    def _set_verify(self, name, value, flags, semantic):
        self._require_mode(name, flags)
        interface, property_id = self._properties[name]
        _check(_call(interface, 4, (_LONG, _LONG, _LONG), property_id, value, flags),
               f"set {name}")
        actual, actual_flags = self._get(name)
        if actual_flags != flags or (flags == _MANUAL and actual != value):
            raise CalibrationError(f"{name}: requested value/mode {value}/{flags}, "
                                   f"driver returned {actual}/{actual_flags}")
        return {"control": name, "semantic": semantic, "requested_value": value,
                "requested_flags": flags, "actual_value": actual,
                "actual_flags": actual_flags, "verified": True}

    def apply_group(self, mode, fixed_exposure_ms=None, fixed_wb_k=None):
        if mode not in ("auto", "fixed"):
            raise CalibrationError("mode must be auto or fixed")
        exposure = "exposure_log2_seconds"
        wb = "white_balance_temperature"
        flag = _AUTO if mode == "auto" else _MANUAL
        self._require_mode(exposure, flag)
        self._require_mode(wb, _AUTO)
        current_exposure, _ = self._get(exposure)
        current_wb, _ = self._get(wb)
        plan = self.resolve_fixed_exposure(fixed_exposure_ms) if mode == "fixed" else None
        value = plan["native_value"] if plan else current_exposure
        # DirectShow Set atomically writes the manual mode and exposure value.
        operations = [self._set_verify(exposure, value, flag, mode + "_exposure"),
                      self._set_verify(wb, current_wb, _AUTO, "auto_white_balance")]
        operations[0]["actual_ms"] = self._exposure_ms(operations[0]["actual_value"])
        if plan:
            operations[0]["requested_ms"] = fixed_exposure_ms
            print(f"[exposure] requested={fixed_exposure_ms:g} ms; "
                  f"driver readback={operations[0]['actual_ms']:g} ms; white_balance=auto")
        return {"mode": mode, "requested": {"exposure_ms": fixed_exposure_ms if plan else "auto",
                                            "white_balance": "auto"},
                "exposure_selection": plan, "operations": operations, "verified": True}

    def restore_initial(self):
        records = []
        for name, (value, flags) in self.initial_native.items():
            if value == UNKNOWN or flags not in (_AUTO, _MANUAL):
                records.append({"control": name, "status": "unknown_original_value"})
                continue
            try:
                current, current_flags = self._get(name)
                if current_flags == flags and (flags == _AUTO or current == value):
                    continue  # Auto values are allowed to evolve; restore the mode.
                records.append(self._set_verify(name, value, flags, "restore_initial"))
            except Exception as exc:
                records.append({"control": name, "status": "failed", "error": str(exc)})
        failed = any(r.get("status") == "failed" for r in records)
        partial = any(r.get("status") == "unknown_original_value" for r in records)
        return {"status": "failed" if failed else "partial" if partial else "restored",
                "operations": records, "finished_at": utc_now()}

    def close(self):
        for interface in reversed(self._interfaces):
            _release(interface)
        self._interfaces.clear()
        _release(self._filter)
        self._filter = None
        if self._com_owned:
            self._ole.CoUninitialize()
            self._com_owned = False
