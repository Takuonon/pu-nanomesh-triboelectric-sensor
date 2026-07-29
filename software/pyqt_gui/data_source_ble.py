from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional
import concurrent.futures

import numpy as np

try:
    from bleak import BleakClient, BleakScanner

    _BLEAK_AVAILABLE = True
except Exception:  # pragma: no cover
    BleakClient = None  # type: ignore[assignment]
    BleakScanner = None  # type: ignore[assignment]
    _BLEAK_AVAILABLE = False

from .data_source import BaseSource


AUDIO_CHAR_UUID = "19B20002-E8F2-537E-4F6C-D104768A1214"  # indicate
CTRL_CHAR_UUID = "19B20001-E8F2-537E-4F6C-D104768A1214"  # write
SERVICE_UUID = "19B20000-E8F2-537E-4F6C-D104768A1214"


CMD_SHUTDOWN = 0xFF


@dataclass(frozen=True)
class BleStatus:
    state: str  # disconnected|connecting|connected|receiving|ready|error
    message: str = ""


class BleAudioSource(BaseSource):
    """Source that provides BLE-received PCM int16 samples as (ch1, ch2).

    Expected flow:
    - connect(): establish the BLE connection in the background.
    - request_capture(seconds): write the start command to CTRL, receive PCM
      over BLE indications, and buffer it.
    - is_capture_done(): return True when reception has completed.
    - get_captured_buffer(): return a buffer with shape (N, 2).

    Accepted receive formats:
    - [seq(2byte LE)] + payload(stereo interleaved: (ch1_i16, ch2_i16) ...)
    - payload(stereo interleaved: (ch1_i16, ch2_i16) ...)

    Notes:
    - The current firmware uses 1 frame = 4 bytes (int16 x 2 channels).
    - For compatibility, mono-like received data is padded with ch2 = 0.
    """

    def __init__(
        self,
        device_name: str = "XIAO-AUDIO",
        service_uuid: str = SERVICE_UUID,
        fs: int = 16000,
        capture_seconds: float = 1.0,
        audio_char_uuid: str = AUDIO_CHAR_UUID,
        ctrl_char_uuid: str = CTRL_CHAR_UUID,
    ):
        self.device_name = device_name
        self.service_uuid = service_uuid
        self.fs = int(fs)
        self.capture_seconds = float(capture_seconds)
        self.audio_char_uuid = audio_char_uuid
        self.ctrl_char_uuid = ctrl_char_uuid

        self._lock = threading.Lock()
        self._status = BleStatus("disconnected")

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        # Keep this typed as Any because bleak may not be installed.
        self._client: Optional[Any] = None

        self._capture_done = threading.Event()
        self._capture_cancel = threading.Event()
        self._captured: Optional[np.ndarray] = None
        self._capture_future: Optional[concurrent.futures.Future] = None

        self._expected_samples: int = int(self.fs * self.capture_seconds)
        self._recv_samples: int = 0
        self._last_error: Optional[str] = None

        # Idle timeout after the final indication if sample count alone is not enough.
        self._notify_idle_timeout_s: float = 0.6

        # Record-then-transfer firmware delays the first indication by the
        # recording duration, so longer captures need a longer first-packet wait.
        self._first_notify_timeout_floor_s: float = 5.0
        self._first_notify_timeout_margin_s: float = 3.0

    # ===== Status API (UI polling) =====
    def get_status(self) -> BleStatus:
        # Keep status consistent with the underlying connection if possible.
        with self._lock:
            st = self._status
            client = self._client

        if client is not None:
            try:
                is_connected = bool(getattr(client, "is_connected", True))
            except Exception:
                is_connected = True
            if (not is_connected) and st.state in ("connected", "ready", "receiving"):
                with self._lock:
                    self._status = BleStatus("disconnected", "Disconnected")
                    return self._status
        return st

    def get_last_error(self) -> Optional[str]:
        with self._lock:
            return self._last_error

    def get_transfer_progress(self) -> tuple[int, int]:
        """(received_samples, expected_samples)"""
        with self._lock:
            return self._recv_samples, self._expected_samples

    # ===== Connection =====
    def connect(self) -> None:
        """Start connecting in background (returns immediately)."""
        if not _BLEAK_AVAILABLE:
            with self._lock:
                self._status = BleStatus("error", "bleak is not installed")
                self._last_error = "bleak is not installed"
            return

        with self._lock:
            # Avoid spamming multiple concurrent connects.
            if self._status.state == "connecting":
                return
            self._status = BleStatus("connecting", f"Connecting to {self.device_name}...")

        # First-time: create the BLE event loop thread.
        if self._thread is None:
            self._thread = threading.Thread(target=self._run_loop_thread, daemon=True)
            self._thread.start()
            return

        # Subsequent calls: re-run connect task on the existing loop.
        if self._loop is not None:
            asyncio.run_coroutine_threadsafe(self._connect_task(), self._loop)

    def _run_loop_thread(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.create_task(self._connect_task())
        self._loop.run_forever()

    def shutdown(self) -> bool:
        """Send the MCU shutdown command (0xFF) via CTRL characteristic.

        Returns True if the command was scheduled to be sent.
        """
        # Ensure loop exists.
        if self._loop is None:
            # If not started yet, attempt connect which will also create the loop.
            self.connect()

        if self._loop is None:
            return False

        asyncio.run_coroutine_threadsafe(self._shutdown_task(), self._loop)
        return True

    async def _shutdown_task(self) -> None:
        # Ensure connected first.
        t0 = time.time()
        while True:
            with self._lock:
                st = self._status.state
                client = self._client
            if st in ("connected", "ready") and client is not None:
                break
            if st == "error":
                return
            if time.time() - t0 > 10.0:
                with self._lock:
                    self._status = BleStatus("error", "Shutdown failed: connect timeout")
                    self._last_error = "Shutdown failed: connect timeout"
                return
            await asyncio.sleep(0.05)

        try:
            payload = bytes([CMD_SHUTDOWN])
            try:
                await client.write_gatt_char(self.ctrl_char_uuid, payload, response=True)
            except Exception:
                await client.write_gatt_char(self.ctrl_char_uuid, payload, response=False)

            with self._lock:
                self._status = BleStatus("connected", "Shutdown command sent")

            # Many firmwares will disconnect immediately after SYSTEM OFF.
            await asyncio.sleep(0.2)
        except Exception as exc:
            with self._lock:
                self._status = BleStatus("error", f"Shutdown failed: {exc}")
                self._last_error = str(exc)

    async def _connect_task(self) -> None:
        try:
            target_name = (self.device_name or "").strip()
            target_svc = (self.service_uuid or "").strip().lower()

            def match(d, ad) -> bool:
                name = (getattr(d, "name", "") or "").strip()
                if target_name and name == target_name:
                    return True
                svc_uuids = getattr(ad, "service_uuids", None) or []
                for u in svc_uuids:
                    if str(u).lower() == target_svc:
                        return True
                return False

            dev = await BleakScanner.find_device_by_filter(match)
            if dev is None:
                raise RuntimeError(f"Device not found (name={target_name!r}, service={target_svc!r})")

            client = BleakClient(dev)
            await client.connect()

            with self._lock:
                self._client = client
                self._status = BleStatus("connected", f"Connected: {self.device_name}")
        except Exception as exc:
            with self._lock:
                self._status = BleStatus("error", f"Connect failed: {exc}")
                self._last_error = str(exc)

    # ===== Capture request/receive =====
    def request_capture(self, seconds: Optional[float] = None) -> bool:
        """Request the MCU to start capture+transfer.

        This implementation writes a one-byte duration command to the CTRL
        characteristic. If the MCU protocol changes, update payload generation here.
        """
        if seconds is not None:
            self.capture_seconds = float(seconds)
        self._expected_samples = int(self.fs * self.capture_seconds)

        # Cancel any in-progress capture so a new one can start.
        if self._capture_future is not None and not self._capture_future.done():
            self._capture_cancel.set()
            try:
                self._capture_future.cancel()
            except Exception:
                pass

        self._capture_done.clear()
        self._capture_cancel.clear()
        with self._lock:
            self._recv_samples = 0
            self._captured = np.zeros((self._expected_samples, 2), dtype=np.float32)
            if self._status.state in ("ready",):
                # Normalize the state so repeated Start presses do not stall _capture_task.
                self._status = BleStatus("connected", "Connected")

            if self._status.state not in ("connected", "ready"):
                # Start the connection first if connect() has not been called.
                if self._thread is None:
                    self.connect()
                # Connection is still pending.
                if self._status.state != "connected":
                    self._status = BleStatus("connecting", "Connecting...")

        # Right after connect(), the event loop may not be initialized yet.
        if self._loop is None:
            deadline = time.time() + 1.0
            while self._loop is None and time.time() < deadline:
                time.sleep(0.01)
        if self._loop is None:
            return False

        self._capture_future = asyncio.run_coroutine_threadsafe(self._capture_task(), self._loop)
        return True

    def cancel_capture(self) -> None:
        self._capture_cancel.set()

    def is_capture_done(self) -> bool:
        return self._capture_done.is_set()

    def get_captured_buffer(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._captured is None else self._captured.copy()

    async def _capture_task(self) -> None:
        # wait until connected (with timeout)
        t0 = time.time()
        while True:
            with self._lock:
                st = self._status.state
                client = self._client
            if st in ("connected", "ready") and client is not None:
                break
            if st == "error":
                return
            if time.time() - t0 > 10.0:
                with self._lock:
                    self._status = BleStatus("error", "Connect timeout")
                    self._last_error = "Connect timeout"
                return
            await asyncio.sleep(0.05)

        # Reconnect if the BLE link has dropped.
        try:
            is_connected = bool(getattr(client, "is_connected", True))
        except Exception:
            is_connected = True
        if not is_connected:
            try:
                await client.connect()
            except Exception as exc:
                with self._lock:
                    self._status = BleStatus("error", f"Reconnect failed: {exc}")
                    self._last_error = str(exc)
                self._capture_done.set()
                return

        with self._lock:
            self._status = BleStatus("receiving", "Receiving...")

        # 2ch (int16,int16) interleaved => 1 frame = 4 bytes
        expected_bytes = int(self._expected_samples) * 4
        if expected_bytes > 0:
            raw_buf = bytearray(expected_bytes)
        else:
            raw_buf = bytearray()
        write_bytes = 0

        # IMPORTANT: _capture_done means "buffer is ready".
        # Do not set it from on_notify; set it only after conversion+copy completes.
        done_requested = False

        # UI progress update (throttled)
        last_report_time = time.time()
        last_rx_time = time.time()
        first_rx = False

        def on_notify(_, data: bytearray):
            nonlocal raw_buf, write_bytes, last_rx_time, first_rx
            nonlocal last_report_time
            nonlocal done_requested

            if self._capture_cancel.is_set() or not data:
                return

            # If already enough, ignore the rest (keeps callback cheap).
            if expected_bytes > 0 and write_bytes >= expected_bytes:
                done_requested = True
                return

            last_rx_time = time.time()
            first_rx = True

            # payload only (some firmwares prepend 2-byte seq)
            # Preferred: 2 + N*4 (stereo), fallback: 2 + N*2 (mono)
            payload_view: memoryview
            if len(data) >= 6 and ((len(data) - 2) % 4 == 0):
                payload_view = memoryview(data)[2:]
            elif len(data) >= 4 and ((len(data) - 2) % 2 == 0):
                payload_view = memoryview(data)[2:]
            else:
                payload_view = memoryview(data)

            # Round down to a frame boundary; stereo is normally 4 bytes per frame.
            if len(payload_view) < 2:
                return
            if (len(payload_view) % 4) != 0:
                # Mono-compatible payloads may only be aligned to 2 bytes.
                if (len(payload_view) % 2) != 0:
                    payload_view = payload_view[: len(payload_view) - 1]
                # Prefer 4-byte alignment when possible.
                payload_view = payload_view[: len(payload_view) - (len(payload_view) % 4)]
            if len(payload_view) < 2:
                return

            # Cap to expected length to avoid unbounded growth
            if expected_bytes > 0:
                remaining_bytes = expected_bytes - write_bytes
                if remaining_bytes <= 0:
                    done_requested = True
                    return
                if len(payload_view) > remaining_bytes:
                    payload_view = payload_view[:remaining_bytes]

                nbytes = len(payload_view)
                raw_buf[write_bytes : write_bytes + nbytes] = payload_view
                write_bytes += nbytes
            else:
                # unknown expected length
                raw_buf.extend(payload_view)

            # Throttled progress update for UI polling
            now = time.time()
            if now - last_report_time > 0.2:
                # report "frames" (samples per channel)
                samples = (write_bytes // 4) if expected_bytes > 0 else (len(raw_buf) // 4)
                with self._lock:
                    self._recv_samples = samples
                last_report_time = now

            if expected_bytes > 0 and write_bytes >= expected_bytes:
                done_requested = True

        notify_started = False
        try:
            await client.start_notify(self.audio_char_uuid, on_notify)
            notify_started = True

            # Start command: 1 byte = recording seconds (e.g., 1/3/5)
            seconds_int = int(round(self.capture_seconds))
            if seconds_int < 1:
                seconds_int = 1
            if seconds_int > 255:
                seconds_int = 255
            payload = bytes([seconds_int])

            # Some BLE stacks reject response=True, so fall back to a write without response.
            try:
                await client.write_gatt_char(self.ctrl_char_uuid, payload, response=True)
            except Exception:
                await client.write_gatt_char(self.ctrl_char_uuid, payload, response=False)

            # Wait until finished/canceled
            start_wait = time.time()
            # If firmware discards warm-up samples before filling targetSamples,
            # transfer starts later than capture_seconds alone would suggest.
            first_notify_timeout_s = max(
                self._first_notify_timeout_floor_s,
                float(self.capture_seconds) + self._first_notify_timeout_margin_s,
            )

            while not done_requested and not self._capture_cancel.is_set():
                # Wait for the first indication before timing out.
                if not first_rx and (time.time() - start_wait) > first_notify_timeout_s:
                    raise RuntimeError("No audio notify received (timeout)")

                # Treat an idle indication stream as complete, even if short.
                if first_rx and (time.time() - last_rx_time) > self._notify_idle_timeout_s:
                    done_requested = True
                    break
                await asyncio.sleep(0.05)

            # Convert once at the end (keeps on_notify cheap)
            view = raw_buf if expected_bytes <= 0 else memoryview(raw_buf)[:write_bytes]
            pcm_i16 = np.frombuffer(view, dtype="<i2")

            # Try stereo first: (ch1, ch2) interleaved.
            frames = int(pcm_i16.shape[0] // 2)
            pcm_i16 = pcm_i16[: frames * 2]
            if frames > 0:
                stereo = pcm_i16.reshape(frames, 2).astype(np.float32, copy=False) / 32768.0
            else:
                stereo = np.zeros((0, 2), dtype=np.float32)

            # If reshape produced nothing but we have data, fall back to mono.
            if stereo.shape[0] == 0 and pcm_i16.shape[0] > 0:
                mono = pcm_i16.astype(np.float32, copy=False) / 32768.0
                stereo = np.zeros((mono.shape[0], 2), dtype=np.float32)
                stereo[:, 0] = mono

            with self._lock:
                if self._captured is not None:
                    take = int(min(stereo.shape[0], self._expected_samples))
                    if take > 0:
                        self._captured[:take, :] = stereo[:take, :]
                    self._recv_samples = take

            # Mark ready after buffer has been written.
            self._capture_done.set()

        except asyncio.CancelledError:
            self._capture_cancel.set()
            with self._lock:
                self._status = BleStatus("connected", "Canceled")
            self._capture_done.set()
            return
        except Exception as exc:
            with self._lock:
                self._status = BleStatus("error", f"Capture failed: {exc}")
                self._last_error = str(exc)
            self._capture_done.set()
            return
        finally:
            if notify_started:
                try:
                    await client.stop_notify(self.audio_char_uuid)
                except Exception:
                    pass

        if self._capture_cancel.is_set():
            with self._lock:
                self._status = BleStatus("connected", "Canceled")
            return

        with self._lock:
            self._status = BleStatus("ready", "Capture ready")

    # ===== BaseSource API =====
    def get_samples(self, n: int) -> np.ndarray:
        # After capture, return the last n samples from the captured buffer.
        with self._lock:
            if self._captured is None:
                return np.zeros((n, 2), dtype=np.float32)
            if n <= 0:
                return np.zeros((0, 2), dtype=np.float32)
            n = min(n, self._captured.shape[0])
            return self._captured[-n:].copy()
