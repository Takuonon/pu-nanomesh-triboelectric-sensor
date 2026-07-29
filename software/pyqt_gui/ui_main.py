# src/ui_main.py
import numpy as np
from PyQt6 import QtWidgets, QtCore
import pyqtgraph as pg

from scipy import signal as sp_signal

from .data_source import BaseSource


def _powerline_bandstop_filter(
    buf: np.ndarray,
    fs: float,
    f0: float = 50.0,
    half_width_hz: float = 1.0,
    order: int = 4,
    rp: float = 0.5,
    rs: float = 60.0,
    repeats: int = 1,
) -> np.ndarray:
    """Remove powerline interference around f0 using a steep IIR bandstop.

    Notes:
    - A super-narrow notch (high-Q) can miss if the interference is slightly off 50 Hz.
      This uses a small stopband (e.g. 49-51 Hz) so removal is more reliable.
    - SOS + sosfiltfilt for numerical stability and zero-phase.
    """
    if buf.ndim != 2 or buf.shape[1] != 2:
        raise ValueError("buffer must be shape (N,2)")
    if buf.shape[0] == 0:
        return buf.copy()

    fs = float(fs)
    f0 = float(f0)
    half_width_hz = float(half_width_hz)
    order = int(order)
    repeats = int(repeats)
    if fs <= 0:
        raise ValueError("fs must be positive")
    if not (0.0 < f0 < fs / 2):
        raise ValueError("f0 must be between 0 and Nyquist")
    if half_width_hz <= 0:
        raise ValueError("half_width_hz must be positive")
    if order <= 0:
        raise ValueError("order must be positive")
    if repeats < 1:
        repeats = 1

    lo = max(0.1, f0 - half_width_hz)
    hi = min(fs / 2 - 0.1, f0 + half_width_hz)
    if not (0.0 < lo < hi < fs / 2):
        raise ValueError("invalid stopband; check fs/f0/half_width_hz")

    sos = sp_signal.iirfilter(
        N=order,
        Wn=[lo, hi],
        rp=float(rp),
        rs=float(rs),
        btype="bandstop",
        ftype="ellip",
        fs=fs,
        output="sos",
    )

    x0 = buf[:, 0].astype(np.float64, copy=False)
    x1 = buf[:, 1].astype(np.float64, copy=False)
    y0 = x0
    y1 = x1
    for _ in range(repeats):
        y0 = sp_signal.sosfiltfilt(sos, y0)
        y1 = sp_signal.sosfiltfilt(sos, y1)
    out = np.stack([y0, y1], axis=1)
    return out.astype(np.float32, copy=False)


def _estimate_sine_amplitude(x: np.ndarray, fs: float, f0: float = 50.0) -> float:
    """Estimate amplitude of a single sine at f0 via least-squares projection."""
    if x.ndim != 1:
        raise ValueError("x must be 1-D")
    n = int(x.size)
    if n == 0:
        return 0.0
    fs = float(fs)
    f0 = float(f0)
    t = np.arange(n, dtype=np.float64) / fs
    w = 2.0 * np.pi * f0
    c = np.cos(w * t)
    s = np.sin(w * t)
    x64 = x.astype(np.float64, copy=False)
    a = (2.0 / n) * float(np.dot(x64, c))
    b = (2.0 / n) * float(np.dot(x64, s))
    return float(np.sqrt(a * a + b * b))


class _UiMode:
    STREAM = "stream"
    BLE_CAPTURE = "ble_capture"


class _CapturePhase:
    IDLE = "idle"
    RECORDING = "recording"
    COMMUNICATING = "communicating"
    DISPLAYING = "displaying"


class MainWindow(QtWidgets.QWidget):
    """
    Main UI.
    - Data comes from BaseSource (Mock / BLE)
    - The UI only needs get_samples(n)
    """

    def __init__(self, source: BaseSource, fs: int = 2000, buffer_seconds: float = 2.0):
        super().__init__()

        self.source = source
        self.fs = fs
        self.buffer_length = int(fs * buffer_seconds)
        self.buffer_seconds = float(buffer_seconds)

        # Detect BLE record-request mode, where data is displayed after reception.
        self._ui_mode = _UiMode.BLE_CAPTURE if hasattr(self.source, "request_capture") else _UiMode.STREAM
        self._capture_phase = _CapturePhase.IDLE
        self._recording_deadline_ms: int | None = None

        self.setWindowTitle("BLE Low-Frequency Audio Monitor")
        self.resize(1000, 700)

        # ========== Top controls ==========
        top_layout = QtWidgets.QHBoxLayout()
        self.connect_btn = QtWidgets.QPushButton("Connect")  # future BLE
        self.shutdown_btn = QtWidgets.QPushButton("Shutdown")
        self.start_btn = QtWidgets.QPushButton("Start")
        self.denoise_btn = QtWidgets.QPushButton("Denoise (50 Hz)")
        self.show_raw_chk = QtWidgets.QCheckBox("Raw")
        self.show_filt_chk = QtWidgets.QCheckBox("Filtered")
        self.filter_label = QtWidgets.QLabel("Filter steepness:")
        self.filter_combo = QtWidgets.QComboBox()
        self.filter_combo.addItems(["Mild", "Normal", "Aggressive", "Extreme"])
        self.save_btn = QtWidgets.QPushButton("Save CSV")

        self.duration_label = QtWidgets.QLabel("Duration (s):")
        self.duration_combo = QtWidgets.QComboBox()
        self.duration_combo.addItems(["1", "2", "3", "4"])
        # Default selection based on initial buffer_seconds.
        default_secs = int(round(self.buffer_seconds))
        if default_secs in (1, 2, 3, 4):
            self.duration_combo.setCurrentText(str(default_secs))
        else:
            self.duration_combo.setCurrentText("1")
            self._apply_duration_seconds(1)

        self.status_label = QtWidgets.QLabel("Status: Idle")

        self.show_raw_chk.setChecked(True)
        self.show_filt_chk.setChecked(False)

        top_layout.addWidget(self.connect_btn)
        top_layout.addWidget(self.shutdown_btn)
        top_layout.addWidget(self.start_btn)
        top_layout.addWidget(self.denoise_btn)
        top_layout.addWidget(self.show_raw_chk)
        top_layout.addWidget(self.show_filt_chk)
        top_layout.addWidget(self.filter_label)
        top_layout.addWidget(self.filter_combo)
        top_layout.addWidget(self.duration_label)
        top_layout.addWidget(self.duration_combo)
        top_layout.addWidget(self.save_btn)
        top_layout.addStretch()
        top_layout.addWidget(self.status_label)

        # ========== Plot area ==========
        self.plot_ch1 = pg.PlotWidget()
        self.plot_ch2 = pg.PlotWidget()

        self.plot_ch1.setTitle("Channel 1")
        self.plot_ch2.setTitle("Channel 2")

        # Keep the y-axis fixed for reproducible visual comparison.
        self.plot_ch1.setYRange(-2, 2)
        self.plot_ch2.setYRange(-2, 2)

        # Time-axis scaling can be adjusted here if needed.
        self.curve_ch1 = self.plot_ch1.plot(pen="y")
        self.curve_ch2 = self.plot_ch2.plot(pen="c")
        # Filtered overlay curves.
        self.curve_ch1_filt = self.plot_ch1.plot(pen=pg.mkPen("r", width=2))
        self.curve_ch2_filt = self.plot_ch2.plot(pen=pg.mkPen("m", width=2))

        # Two-channel data buffer.
        self.buffer = np.zeros((self.buffer_length, 2))
        self.filtered_buffer: np.ndarray | None = None

        # ===== DC-offset removal for ADC ch1 =====
        # If the Arduino-side analog input is biased around 1.65 V, center it
        # for visualization by subtracting the capture-wide mean.
        # Set _ch1_dc_auto=False and _ch1_dc_offset_fixed for a fixed offset.
        self._ch1_dc_auto = True
        self._ch1_dc_offset_fixed = 0.0
        # Powerline-noise removal defaults to a 49-51 Hz bandstop.
        self._powerline_hz: float = 50.0
        self._powerline_half_width_hz: float = 1.0
        self._powerline_order: int = 4
        self._powerline_rp: float = 0.5
        self._powerline_rs: float = 60.0
        self._powerline_repeats: int = 1

        # UI preset
        self._set_filter_preset("Normal")
        self.filter_combo.setCurrentText("Normal")

        # Overall layout.
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.plot_ch1)
        main_layout.addWidget(self.plot_ch2)

        # Timer for real-time updates.
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._on_timer)

        # Keep polling BLE status after Connect.
        self.status_timer: QtCore.QTimer | None = None
        if self._ui_mode == _UiMode.BLE_CAPTURE:
            self.status_timer = QtCore.QTimer()
            self.status_timer.timeout.connect(self._update_ble_status_label)
            self.status_timer.start(200)

        # Signal connections.
        self.start_btn.clicked.connect(self.start_acquire)
        self.denoise_btn.clicked.connect(self.apply_denoise)
        self.save_btn.clicked.connect(self.save_csv)
        # Connect button delegates to BLE sources that implement connect().
        self.connect_btn.clicked.connect(self.on_connect_clicked)
        self.shutdown_btn.clicked.connect(self.on_shutdown_clicked)
        self.duration_combo.currentTextChanged.connect(self._on_duration_changed)
        self.filter_combo.currentTextChanged.connect(self._on_filter_preset_changed)
        self.show_raw_chk.toggled.connect(self._refresh_plot)
        self.show_filt_chk.toggled.connect(self._refresh_plot)

        # Initial state.
        self._acquiring = False

        # Initial plot.
        self._refresh_plot()

        if self._ui_mode == _UiMode.BLE_CAPTURE:
            # Disable capture controls until the BLE connection is ready.
            self.start_btn.setEnabled(False)
            self.shutdown_btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            self.duration_label.setEnabled(False)
            self.duration_combo.setEnabled(False)
            self.filter_label.setEnabled(False)
            self.filter_combo.setEnabled(False)
            self._update_ble_status_label()
        else:
            # Streaming mode doesn't support shutdown.
            self.shutdown_btn.setEnabled(False)

    # ========== Button handlers ==========
    def start_acquire(self):
        if self._acquiring:
            return

        # Clear the previous capture before every Start.
        self.buffer = np.zeros((self.buffer_length, 2), dtype=np.float32)
        self.filtered_buffer = None
        self.show_raw_chk.setChecked(True)
        self.show_filt_chk.setChecked(False)
        self.status_label.setToolTip("")
        t = np.arange(self.buffer_length) / self.fs
        self.curve_ch1.setData(t, self.buffer[:, 0])
        self.curve_ch2.setData(t, self.buffer[:, 1])
        self.curve_ch1_filt.setData([], [])
        self.curve_ch2_filt.setData([], [])

        self._acquiring = True
        self.duration_combo.setEnabled(False)

        # Streaming sources update continuously.
        if self._ui_mode == _UiMode.STREAM:
            self.status_label.setText("Status: Running")
            self.timer.start(50)
            return

        # BLE capture: request -> recording countdown -> transfer -> display.
        error = False

        if not hasattr(self.source, "request_capture"):
            self.status_label.setText("Status: Error (source has no request_capture)")
            error = True
        else:
            ok = bool(self.source.request_capture(self.buffer_seconds))  # type: ignore[attr-defined]
            if not ok:
                self.status_label.setText("Status: Error (BLE loop not ready)")
                error = True

        if error:
            self._acquiring = False
            self.duration_combo.setEnabled(True)
            return

        now_ms = int(QtCore.QDateTime.currentMSecsSinceEpoch())
        self._recording_deadline_ms = now_ms + int(self.buffer_seconds * 1000)
        self._capture_phase = _CapturePhase.RECORDING
        self.status_label.setText(f"Status: Recording ({self.buffer_seconds:.1f}s left)")
        self.timer.start(50)

    def apply_denoise(self):
        """Apply 50Hz powerline removal filter and overlay the result."""
        try:
            self.filtered_buffer = self._compute_notch_filtered(self.buffer)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Denoise", f"Denoise failed: {exc}")
            return

        # Estimate 50 Hz attenuation to help diagnose subtle visual changes.
        try:
            raw_a1 = _estimate_sine_amplitude(self.buffer[:, 0], fs=float(self.fs), f0=self._powerline_hz)
            raw_a2 = _estimate_sine_amplitude(self.buffer[:, 1], fs=float(self.fs), f0=self._powerline_hz)
            fil_a1 = _estimate_sine_amplitude(self.filtered_buffer[:, 0], fs=float(self.fs), f0=self._powerline_hz)
            fil_a2 = _estimate_sine_amplitude(self.filtered_buffer[:, 1], fs=float(self.fs), f0=self._powerline_hz)
            msg = f"50Hz amp ch1 {raw_a1:.3g}->{fil_a1:.3g}, ch2 {raw_a2:.3g}->{fil_a2:.3g}"
            self.status_label.setToolTip(msg)
            if not self._acquiring:
                self.status_label.setText(f"Status: Denoised ({msg})")
        except Exception:
            pass

        # Enable the filtered overlay while keeping the raw waveform visible.
        self.show_raw_chk.setChecked(True)
        self.show_filt_chk.setChecked(True)
        self._refresh_plot()

    def on_connect_clicked(self):
        """Connect to the BLE device (BLE mode only)."""
        if hasattr(self.source, "connect"):
            try:
                self.source.connect()  # type: ignore[attr-defined]
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Connect", f"Connect failed: {exc}")
                return
            self.status_label.setText("Status: Connecting...")
            return

        QtWidgets.QMessageBox.information(self, "Connect", "This source does not support BLE connect().")

    def on_shutdown_clicked(self):
        """Send shutdown command to the BLE device (BLE mode only)."""
        if self._ui_mode != _UiMode.BLE_CAPTURE:
            QtWidgets.QMessageBox.information(self, "Shutdown", "Shutdown is available in BLE mode only.")
            return

        if self._acquiring:
            QtWidgets.QMessageBox.warning(
                self, "Shutdown", "Capturing/transfer is in progress. Stop or wait before shutdown."
            )
            return

        if not hasattr(self.source, "shutdown"):
            QtWidgets.QMessageBox.information(self, "Shutdown", "This source does not support shutdown().")
            return

        ret = QtWidgets.QMessageBox.question(
            self,
            "Shutdown",
            "This will put the device into low-power System OFF.\n"
            "Wake-up requires pressing RESET or power-cycling the board.\n\n"
            "Continue?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if ret != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        try:
            ok = bool(self.source.shutdown())  # type: ignore[attr-defined]
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Shutdown", f"Shutdown failed: {exc}")
            return

        if ok:
            self.status_label.setText("Status: Shutdown command sent")
            # Disable immediately; status timer will update once disconnected.
            self.shutdown_btn.setEnabled(False)
        else:
            self.status_label.setText("Status: Error (shutdown not scheduled)")

    def save_csv(self):
        """Save the current buffer as CSV (ch1,ch2)."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save CSV", "data.csv", "CSV Files (*.csv)")
        if path:
            np.savetxt(path, self.buffer, delimiter=",", header="ch1,ch2", comments="")
            self.status_label.setText(f"Status: Saved to {path}")

    def closeEvent(self, event):  # noqa: N802 (Qt naming)
        if self.timer.isActive():
            self.timer.stop()
        if self.status_timer is not None and self.status_timer.isActive():
            self.status_timer.stop()
        super().closeEvent(event)

    def _on_duration_changed(self, text: str):
        if self._acquiring:
            return
        try:
            seconds = int(text)
        except Exception:
            return
        if seconds not in (1, 2, 3, 4):
            return
        self._apply_duration_seconds(seconds)

    def _apply_duration_seconds(self, seconds: int):
        # Update buffer window length and reset display.
        self.buffer_seconds = float(seconds)
        self.buffer_length = int(self.fs * self.buffer_seconds)
        self.buffer = np.zeros((self.buffer_length, 2), dtype=np.float32)
        self.filtered_buffer = None
        t = np.arange(self.buffer_length) / self.fs
        self.curve_ch1.setData(t, self.buffer[:, 0])
        self.curve_ch2.setData(t, self.buffer[:, 1])
        self.curve_ch1_filt.setData([], [])
        self.curve_ch2_filt.setData([], [])
        self._refresh_plot()

    def _set_filter_preset(self, name: str):
        """Set filter params from a UI preset name."""
        # All presets keep the stopband wide enough to tolerate small mains drift.
        self._powerline_half_width_hz = 1.0
        self._powerline_rp = 0.5

        if name == "Mild":
            self._powerline_order = 2
            self._powerline_rs = 40.0
            self._powerline_repeats = 1
            self._powerline_half_width_hz = 1.5
        elif name == "Aggressive":
            self._powerline_order = 6
            self._powerline_rs = 80.0
            self._powerline_repeats = 1
        elif name == "Extreme":
            self._powerline_order = 6
            self._powerline_rs = 100.0
            self._powerline_repeats = 2
        else:  # "Normal" or unknown
            self._powerline_order = 4
            self._powerline_rs = 60.0
            self._powerline_repeats = 1

    def _on_filter_preset_changed(self, text: str):
        if self._acquiring:
            return
        self._set_filter_preset(text)

        # If filtered view is enabled or already computed, recompute immediately.
        if self.show_filt_chk.isChecked() or self.filtered_buffer is not None:
            try:
                self.filtered_buffer = self._compute_notch_filtered(self.buffer)
            except Exception:
                self.filtered_buffer = None
        self._refresh_plot()

    def _compute_notch_filtered(self, buf: np.ndarray) -> np.ndarray:
        return _powerline_bandstop_filter(
            buf,
            fs=float(self.fs),
            f0=self._powerline_hz,
            half_width_hz=self._powerline_half_width_hz,
            order=self._powerline_order,
            rp=self._powerline_rp,
            rs=self._powerline_rs,
            repeats=self._powerline_repeats,
        )

    def _refresh_plot(self):
        """Refresh plot visibility/data based on current state and toggles."""
        t = np.arange(self.buffer_length) / self.fs

        show_raw = bool(self.show_raw_chk.isChecked())
        show_filt = bool(self.show_filt_chk.isChecked())

        # Raw waveform.
        self.curve_ch1.setVisible(show_raw)
        self.curve_ch2.setVisible(show_raw)
        if show_raw:
            self.curve_ch1.setData(t, self.buffer[:, 0])
            self.curve_ch2.setData(t, self.buffer[:, 1])

        # Filtered waveform.
        if show_filt and self.filtered_buffer is None:
            try:
                self.filtered_buffer = self._compute_notch_filtered(self.buffer)
            except Exception:
                self.filtered_buffer = None
                show_filt = False
                self.show_filt_chk.setChecked(False)

        self.curve_ch1_filt.setVisible(show_filt)
        self.curve_ch2_filt.setVisible(show_filt)
        if show_filt and self.filtered_buffer is not None:
            self.curve_ch1_filt.setData(t, self.filtered_buffer[:, 0])
            self.curve_ch2_filt.setData(t, self.filtered_buffer[:, 1])

    def _update_ble_status_label(self):
        """Update status label from BLE source state when not actively acquiring."""
        if self._ui_mode != _UiMode.BLE_CAPTURE:
            return
        if self._acquiring:
            return
        if self._capture_phase == _CapturePhase.DISPLAYING:
            # Keep the final message until next Start.
            return

        if not hasattr(self.source, "get_status"):
            return

        try:
            st = self.source.get_status()  # type: ignore[attr-defined]
        except Exception:
            return

        state = getattr(st, "state", "")
        msg = getattr(st, "message", "")

        if state == "connected":
            self.status_label.setText("Status: Connected")
            # Once connected, lock the Connect button.
            self.connect_btn.setEnabled(False)
            self.shutdown_btn.setEnabled(True)
            self.start_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            self.duration_label.setEnabled(True)
            self.duration_combo.setEnabled(True)
            self.filter_label.setEnabled(True)
            self.filter_combo.setEnabled(True)
        elif state == "ready":
            self.status_label.setText("Status: Ready")
            self.connect_btn.setEnabled(False)
            self.shutdown_btn.setEnabled(True)
            self.start_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            self.duration_label.setEnabled(True)
            self.duration_combo.setEnabled(True)
            self.filter_label.setEnabled(True)
            self.filter_combo.setEnabled(True)
        elif state == "receiving":
            self.status_label.setText("Status: Transferring")
            # Treat receiving as connected; active captures are guarded by _acquiring.
            self.connect_btn.setEnabled(False)
            self.shutdown_btn.setEnabled(True)
            self.start_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            self.duration_label.setEnabled(True)
            self.duration_combo.setEnabled(True)
            self.filter_label.setEnabled(True)
            self.filter_combo.setEnabled(True)
        elif state == "disconnected":
            self.status_label.setText("Status: Disconnected")
            if msg:
                self.status_label.setToolTip(msg)
            self.connect_btn.setEnabled(True)
            self.shutdown_btn.setEnabled(False)
            self.start_btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            self.duration_label.setEnabled(False)
            self.duration_combo.setEnabled(False)
            self.filter_label.setEnabled(False)
            self.filter_combo.setEnabled(False)
        elif state == "connecting":
            # Keep it short but informative
            self.status_label.setText("Status: Connecting...")
            if msg:
                self.status_label.setToolTip(msg)
            self.start_btn.setEnabled(False)
            self.shutdown_btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            self.duration_label.setEnabled(False)
            self.duration_combo.setEnabled(False)
            self.filter_label.setEnabled(False)
            self.filter_combo.setEnabled(False)
        elif state == "error":
            self.status_label.setText(f"Status: Error ({msg})" if msg else "Status: Error")
            self.start_btn.setEnabled(False)
            self.shutdown_btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            self.duration_label.setEnabled(False)
            self.duration_combo.setEnabled(False)
            self.filter_label.setEnabled(False)
            self.filter_combo.setEnabled(False)
        else:
            self.status_label.setText("Status: Idle")
            self.start_btn.setEnabled(False)
            self.shutdown_btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            self.duration_label.setEnabled(False)
            self.duration_combo.setEnabled(False)
            self.filter_label.setEnabled(False)
            self.filter_combo.setEnabled(False)

    # ========== Plot updates called from the timer ==========
    def update_plot(self):
        # Number of samples fetched per update; adjust if the source rate changes.
        n = int(self.fs * 0.05)  # 50 ms worth of samples.
        data = self.source.get_samples(n)  # shape = (n,2)

        # Drop old samples and append the newest data.
        if data.shape[1] != 2:
            raise ValueError("Source must provide 2-channel data (shape = (n,2))")

        self.buffer = np.vstack([self.buffer[n:], data])

        # Recompute the filtered view when the overlay is enabled.
        if self.show_filt_chk.isChecked():
            try:
                self.filtered_buffer = self._compute_notch_filtered(self.buffer)
            except Exception:
                self.filtered_buffer = None

        self._refresh_plot()

    # ========== Timer dispatcher ==========
    def _on_timer(self):
        if self._ui_mode == _UiMode.STREAM:
            self.update_plot()
            return

        # BLE capture mode
        if not self._acquiring:
            return

        now_ms = int(QtCore.QDateTime.currentMSecsSinceEpoch())

        if self._capture_phase == _CapturePhase.RECORDING:
            if self._recording_deadline_ms is None:
                self._recording_deadline_ms = now_ms
            remaining_ms = max(0, self._recording_deadline_ms - now_ms)
            remaining_s = remaining_ms / 1000.0
            self.status_label.setText(f"Status: Recording ({remaining_s:.1f}s left)")
            if remaining_ms <= 0:
                self._capture_phase = _CapturePhase.COMMUNICATING
                self.status_label.setText("Status: Transferring")
            return

        if self._capture_phase == _CapturePhase.COMMUNICATING:
            done = False
            if hasattr(self.source, "is_capture_done"):
                try:
                    done = bool(self.source.is_capture_done())  # type: ignore[attr-defined]
                except Exception:
                    done = False

            if not done:
                # Show transfer progress if the source exposes it.
                if hasattr(self.source, "get_transfer_progress"):
                    try:
                        recvd, expected = self.source.get_transfer_progress()  # type: ignore[attr-defined]
                        self.status_label.setText(f"Status: Transferring ({recvd}/{expected} samples)")
                    except Exception:
                        pass
                return

            # Capture complete: fetch and display the buffer.
            buf = None
            if hasattr(self.source, "get_captured_buffer"):
                try:
                    buf = self.source.get_captured_buffer()  # type: ignore[attr-defined]
                except Exception:
                    buf = None

            if buf is None or not isinstance(buf, np.ndarray) or buf.size == 0:
                self.status_label.setText("Status: Error (no captured data)")
                self._acquiring = False
                self.timer.stop()
                self.duration_combo.setEnabled(True)
                return

            if buf.ndim != 2 or buf.shape[1] != 2:
                self.status_label.setText("Status: Error (captured shape mismatch)")
                self._acquiring = False
                self.timer.stop()
                self.duration_combo.setEnabled(True)
                return

            if buf.shape[0] != self.buffer_length:
                # Fit to the UI display length; pad short buffers with zeros.
                if buf.shape[0] > self.buffer_length:
                    buf = buf[-self.buffer_length :]
                else:
                    pad = np.zeros((self.buffer_length - buf.shape[0], 2), dtype=buf.dtype)
                    buf = np.vstack([pad, buf])

            # Center ch1 (ADC) by removing DC offset.
            try:
                buf = buf.copy()
                if bool(self._ch1_dc_auto):
                    dc = float(np.mean(buf[:, 0]))
                else:
                    dc = float(self._ch1_dc_offset_fixed)
                buf[:, 0] = buf[:, 0] - dc
            except Exception:
                # Visualization should continue even if centering fails.
                pass

            self.buffer = buf
            # Refresh the plot after capture; apply the notch if requested.
            if self.show_filt_chk.isChecked():
                try:
                    self.filtered_buffer = self._compute_notch_filtered(self.buffer)
                except Exception:
                    self.filtered_buffer = None
            self._refresh_plot()

            self._capture_phase = _CapturePhase.DISPLAYING
            self.status_label.setText("Status: Displaying")
            self._acquiring = False
            self.timer.stop()
            self.duration_combo.setEnabled(True)
            return
