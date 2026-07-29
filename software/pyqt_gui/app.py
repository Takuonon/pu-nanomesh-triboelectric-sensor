# src/app.py
import sys
from PyQt6 import QtWidgets

from .data_source_ble import BleAudioSource
from .ui_main import MainWindow


def main():
    app = QtWidgets.QApplication(sys.argv)

    # BLE mode is the default; mock input is not used in this release.
    # Current XIAO nRF52840 firmware: 2 channels, 8 kHz storage, 1-4 s captures.
    fs = 8000
    buffer_seconds = 1.0
    source = BleAudioSource(
        device_name="XIAO-AUDIO",
        service_uuid="19B20000-E8F2-537E-4F6C-D104768A1214",
        fs=fs,
        capture_seconds=buffer_seconds,
    )
    # Start BLE connection automatically in the background.
    source.connect()

    window = MainWindow(source=source, fs=fs, buffer_seconds=buffer_seconds)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
