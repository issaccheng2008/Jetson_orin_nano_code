"""Vision <-> MainBoard 通讯协议 V2 (P1) 实现.

参考: docs/motor_protocol_v2_p1.md / docs/motor_protocol_v2_p1.pdf

设计目标:
- 同时兼容 OpenMV (MicroPython) 与 CPython (仿真侧).
- 单文件零外部依赖.
- 已有接口 build_heartbeat / build_line_ctrl 字节级保持向后兼容.
"""

try:
	import ustruct as struct  # type: ignore
except Exception:
	import struct  # type: ignore

import time


def _ticks_ms():
	try:
		return int(time.ticks_ms())  # type: ignore
	except Exception:
		return int(time.time() * 1000)


def _ticks_diff(a, b):
	try:
		return int(time.ticks_diff(a, b))  # type: ignore
	except Exception:
		return int(a - b)


def quantize_to_step(v, step):
	if step <= 1:
		return int(v)
	return int(round(float(v) / float(step)) * float(step))


def _clamp_i16(v):
	if v < -32768:
		return -32768
	if v > 32767:
		return 32767
	return int(v)


def _crc16_ccitt_false(data):
	crc = 0xFFFF
	for b in data:
		crc ^= (b & 0xFF) << 8
		for _ in range(8):
			if crc & 0x8000:
				crc = ((crc << 1) ^ 0x1021) & 0xFFFF
			else:
				crc = (crc << 1) & 0xFFFF
	return crc


SOF1 = 0x55
SOF2 = 0xAA

MSG_HEARTBEAT = 0x01
MSG_LINE_CTRL = 0x02
MSG_QR_EVENT = 0x10
MSG_OBSTACLE_EVENT = 0x11
MSG_MODE_SWITCH_REQ = 0x12
MSG_ACK = 0x80
MSG_ROBOT_STATE = 0x81
MSG_ESTOP = 0xE0

FLAG_ACK_REQ = 0x01
FLAG_IS_ACK = 0x02

EVENT_TYPES = (MSG_QR_EVENT, MSG_OBSTACLE_EVENT, MSG_MODE_SWITCH_REQ, MSG_ESTOP)

# ACK 负载约定 (PDF 未明确定义, 取常用最小集): orig_seq(1B) + orig_msg_type(1B) + code(1B)
ACK_PAYLOAD_LEN = 3

# 帧最小长度: SOF(2) + VER(1) + MSG(1) + FLAGS(1) + SEQ(1) + TS_MS(4) + LEN(1) + CRC(2) = 13
_MIN_FRAME_LEN = 13
_HEADER_BEFORE_PAYLOAD = 11  # SOF..LEN
_BODY_OFFSET = 2  # CRC 校验起点 (跳过 SOF1/SOF2)


class VisionProtocolV2:
	"""帧打包器, 维持单调递增 seq."""

	SOF1 = SOF1
	SOF2 = SOF2
	MSG_HEARTBEAT = MSG_HEARTBEAT
	MSG_LINE_CTRL = MSG_LINE_CTRL

	def __init__(self, version=2):
		self.version = int(version) & 0xFF
		self.seq = 0

	def _next_seq(self):
		s = self.seq
		self.seq = (self.seq + 1) & 0xFF
		return s

	def build_frame(self, msg_type, payload=b"", flags=0, seq=None, ts_ms=None):
		if ts_ms is None:
			ts_ms = _ticks_ms()
		if seq is None:
			seq = self._next_seq()
		if payload is None:
			payload = b""

		header = struct.pack(
			"<BBBBIB",
			self.version,
			int(msg_type) & 0xFF,
			int(flags) & 0xFF,
			int(seq) & 0xFF,
			int(ts_ms) & 0xFFFFFFFF,
			len(payload) & 0xFF,
		)
		body = header + bytes(payload)
		crc = _crc16_ccitt_false(body)
		return bytes([SOF1, SOF2]) + body + struct.pack("<H", crc)

	def build_heartbeat(self, mode_u8, ts_ms=None):
		payload = struct.pack("<B", int(mode_u8) & 0xFF)
		return self.build_frame(MSG_HEARTBEAT, payload, ts_ms=ts_ms)

	def build_line_ctrl(
		self,
		mode_u8,
		conf_u8,
		lost_u8,
		route_u8,
		ex_mm_i16=0,
		ang_cdeg_i16=0,
		v_cmd_mmps_i16=0,
		w_cmd_mradps_i16=0,
		ts_ms=None,
	):
		# 12 字节 payload, 顺序参考 docs/motor_protocol_v2_p1.md §4
		payload = struct.pack(
			"<BBBBhhhh",
			int(mode_u8) & 0xFF,
			int(conf_u8) & 0xFF,
			int(lost_u8) & 0xFF,
			int(route_u8) & 0xFF,
			_clamp_i16(ex_mm_i16),
			_clamp_i16(ang_cdeg_i16),
			_clamp_i16(v_cmd_mmps_i16),
			_clamp_i16(w_cmd_mradps_i16),
		)
		return self.build_frame(MSG_LINE_CTRL, payload, ts_ms=ts_ms)

	def build_event(self, msg_type, payload=b"", request_ack=True, seq=None, ts_ms=None):
		flags = FLAG_ACK_REQ if request_ack else 0
		return self.build_frame(msg_type, payload, flags=flags, seq=seq, ts_ms=ts_ms)

	def build_qr_event(self, action_id, request_ack=True):
		return self.build_event(MSG_QR_EVENT, struct.pack("<B", int(action_id) & 0xFF), request_ack)

	def build_obstacle_event(self, present_u8, distance_cm_u8=0, request_ack=True):
		payload = struct.pack("<BB", int(present_u8) & 0xFF, int(distance_cm_u8) & 0xFF)
		return self.build_event(MSG_OBSTACLE_EVENT, payload, request_ack)

	def build_mode_switch_req(self, target_mode_u8, request_ack=True):
		return self.build_event(MSG_MODE_SWITCH_REQ, struct.pack("<B", int(target_mode_u8) & 0xFF), request_ack)

	def build_estop(self, reason_code=0, request_ack=True):
		return self.build_event(MSG_ESTOP, struct.pack("<B", int(reason_code) & 0xFF), request_ack)

	def build_ack(self, orig_seq, orig_msg_type, code=0, ts_ms=None):
		payload = struct.pack(
			"<BBB",
			int(orig_seq) & 0xFF,
			int(orig_msg_type) & 0xFF,
			int(code) & 0xFF,
		)
		return self.build_frame(MSG_ACK, payload, flags=FLAG_IS_ACK, ts_ms=ts_ms)


class Frame(object):
	"""解析后的帧对象 (避免 namedtuple 在 MicroPython 上的不一致)."""

	def __init__(self, ver, msg_type, flags, seq, ts_ms, payload):
		self.ver = ver
		self.msg_type = msg_type
		self.flags = flags
		self.seq = seq
		self.ts_ms = ts_ms
		self.payload = payload

	def is_ack(self):
		return bool(self.flags & FLAG_IS_ACK)

	def needs_ack(self):
		return bool(self.flags & FLAG_ACK_REQ)

	def __repr__(self):
		return "Frame(msg=0x%02X, seq=%d, len=%d)" % (self.msg_type, self.seq, len(self.payload))


class StreamParser(object):
	"""流式状态机, 把任意分片字节流切成完整帧.

	用法:
		parser = StreamParser()
		for frame in parser.feed(rx_bytes):
			handle(frame)
	"""

	def __init__(self, max_payload=255):
		self._buf = bytearray()
		self.max_payload = int(max_payload)
		self.frames_ok = 0
		self.crc_errors = 0
		self.discarded_bytes = 0

	def feed(self, chunk):
		if chunk:
			self._buf.extend(chunk)
		out = []
		while True:
			f = self._try_one()
			if f is None:
				break
			out.append(f)
		return out

	def reset(self):
		self._buf = bytearray()

	def _try_one(self):
		buf = self._buf
		n = len(buf)
		if n < _MIN_FRAME_LEN:
			return None

		# 找第一个 SOF1+SOF2
		i = 0
		while i < n - 1:
			if buf[i] == SOF1 and buf[i + 1] == SOF2:
				break
			i += 1
		else:
			# 整段都没有 SOF, 全部丢掉, 但保留最后一个字节 (可能下次拼成 SOF)
			drop = max(0, n - 1)
			if drop > 0:
				self.discarded_bytes += drop
				del buf[:drop]
			return None

		if i > 0:
			self.discarded_bytes += i
			del buf[:i]
			n = len(buf)
			if n < _MIN_FRAME_LEN:
				return None

		# 现在 buf[0..1] = SOF1 SOF2, buf[2..10] = VER..LEN
		plen = buf[10]
		if plen > self.max_payload:
			# LEN 不合理, 这一对 SOF 是误识别, 跳过 1 字节继续找
			self.discarded_bytes += 1
			del buf[:1]
			return None

		total = 2 + 9 + plen + 2  # SOF + header + payload + CRC
		if n < total:
			return None

		ver = buf[2]
		msg_type = buf[3]
		flags = buf[4]
		seq = buf[5]
		ts_ms = struct.unpack("<I", bytes(buf[6:10]))[0]

		body_end = 11 + plen
		body = bytes(buf[2:body_end])  # VER..PAYLOAD
		crc_recv = struct.unpack("<H", bytes(buf[body_end:body_end + 2]))[0]
		crc_calc = _crc16_ccitt_false(body)

		del buf[:total]

		if crc_recv != crc_calc:
			self.crc_errors += 1
			return None

		self.frames_ok += 1
		# body layout: VER(1) MSG(1) FLAGS(1) SEQ(1) TS_MS(4) LEN(1) PAYLOAD(plen) -> payload starts at index 9
		payload = bytes(body[9:]) if plen > 0 else b""
		return Frame(ver, msg_type, flags, seq, ts_ms, payload)


def parse_line_ctrl(payload):
	if len(payload) < 12:
		return None
	mode, conf, lost, route, ex, ang, v_cmd, w_cmd = struct.unpack(
		"<BBBBhhhh", bytes(payload[:12])
	)
	return {
		"mode": mode,
		"conf": conf,
		"lost": lost,
		"route": route,
		"ex_mm": ex,
		"ang_cdeg": ang,
		"v_cmd_mmps": v_cmd,
		"w_cmd_mradps": w_cmd,
	}


def parse_heartbeat(payload):
	if len(payload) < 1:
		return None
	return {"mode": payload[0]}


def parse_ack(payload):
	if len(payload) < ACK_PAYLOAD_LEN:
		return None
	return {
		"ack_seq": payload[0],
		"ack_msg_type": payload[1],
		"code": payload[2],
	}


class PendingAcks(object):
	"""事件帧 ACK 等待 + 超时重发.

	约束 (PDF §6): 80ms 超时, 最多 3 次重发.

	用法:
		pending = PendingAcks(send_func, timeout_ms=80, retry_max=3)
		pending.send(frame_bytes, seq, msg_type)
		# 主循环周期调用:
		pending.tick()
		# 收到 ACK 时:
		pending.on_ack(ack_seq, ack_msg_type)
	"""

	def __init__(self, send_func, timeout_ms=80, retry_max=3):
		self._send = send_func
		self._timeout_ms = int(timeout_ms)
		self._retry_max = int(retry_max)
		self._pending = {}
		self.dropped = 0

	def send(self, frame_bytes, seq, msg_type):
		try:
			self._send(frame_bytes)
		except Exception:
			pass
		self._pending[(int(seq) & 0xFF, int(msg_type) & 0xFF)] = {
			"bytes": frame_bytes,
			"last_ts": _ticks_ms(),
			"retries": 0,
		}

	def on_ack(self, ack_seq, ack_msg_type):
		key = (int(ack_seq) & 0xFF, int(ack_msg_type) & 0xFF)
		if key in self._pending:
			del self._pending[key]
			return True
		return False

	def tick(self):
		now = _ticks_ms()
		dead = []
		for key in list(self._pending.keys()):
			ent = self._pending[key]
			if _ticks_diff(now, ent["last_ts"]) >= self._timeout_ms:
				if ent["retries"] >= self._retry_max:
					dead.append(key)
				else:
					ent["retries"] += 1
					ent["last_ts"] = now
					try:
						self._send(ent["bytes"])
					except Exception:
						pass
		for key in dead:
			del self._pending[key]
			self.dropped += 1

	def pending_count(self):
		return len(self._pending)


class SafetyMonitor(object):
	"""超时安全哨兵 (PDF §7-8).

	视觉侧用法 (本机自检发送是否按节奏):
		mon = SafetyMonitor(ctrl_period_ms=100, heartbeat_period_ms=100)
		# 每次发完后:
		mon.mark_sent_ctrl()
		mon.mark_sent_hb()
		# 周期检查:
		hi = mon.check_send_health()  # ctrl_misses / hb_misses 累加

	电机侧用法 (监控接收是否在超时窗内):
		state = mon.assess_receive(last_rx_ms)  # SAFE_OK / SAFE_STOP / LOST_RECOVERY
	"""

	SAFE_OK = 0
	SAFE_STOP = 1
	LOST_RECOVERY = 2

	def __init__(
		self,
		ctrl_period_ms=100,
		heartbeat_period_ms=100,
		safe_stop_ms=300,
		lost_recovery_ms=800,
	):
		self.ctrl_period_ms = int(ctrl_period_ms)
		self.heartbeat_period_ms = int(heartbeat_period_ms)
		self.safe_stop_ms = int(safe_stop_ms)
		self.lost_recovery_ms = int(lost_recovery_ms)
		self._last_ctrl_ms = None
		self._last_hb_ms = None
		self.ctrl_misses = 0
		self.hb_misses = 0

	def mark_sent_ctrl(self, ts_ms=None):
		self._last_ctrl_ms = _ticks_ms() if ts_ms is None else int(ts_ms)

	def mark_sent_hb(self, ts_ms=None):
		self._last_hb_ms = _ticks_ms() if ts_ms is None else int(ts_ms)

	def check_send_health(self):
		now = _ticks_ms()
		ctrl_age = None if self._last_ctrl_ms is None else _ticks_diff(now, self._last_ctrl_ms)
		hb_age = None if self._last_hb_ms is None else _ticks_diff(now, self._last_hb_ms)
		if ctrl_age is not None and ctrl_age > 2 * self.ctrl_period_ms:
			self.ctrl_misses += 1
		if hb_age is not None and hb_age > 2 * self.heartbeat_period_ms:
			self.hb_misses += 1
		return {
			"ctrl_age_ms": ctrl_age,
			"hb_age_ms": hb_age,
			"ctrl_misses": self.ctrl_misses,
			"hb_misses": self.hb_misses,
		}

	def assess_receive(self, last_rx_ms):
		if last_rx_ms is None:
			return self.SAFE_STOP
		age = _ticks_diff(_ticks_ms(), int(last_rx_ms))
		if age > self.lost_recovery_ms:
			return self.LOST_RECOVERY
		if age > self.safe_stop_ms:
			return self.SAFE_STOP
		return self.SAFE_OK
