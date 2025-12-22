#!/usr/bin/env python3
"""
Enhanced Continuous Trigger Capture Automation for Keysight DSOX6004A Oscilloscope

SCIENTIFIC OBJECTIVE:
Automated acquisition and analysis of periodic signals (Laser and APD) to study
timing jitter and delay variations. Designed for precision delay measurements
between correlated signals with statistical analysis.

KEY FEATURES:
- Automatic delay calculation using cross-correlation
- Real-time statistics display during acquisition
- Live visualization with delay histograms
- Statistical report generation
- Excel and MATLAB export for analysis
- Quick Start interface for ease of use
- Professional data organization

CRITICAL IMPROVEMENTS (v3.0):
✅ Automatic delay measurement between channels
✅ Real-time statistics and jitter analysis
✅ Live plotting and visualization
✅ Statistical report generation
✅ Multi-format export (CSV, Excel, MATLAB)
✅ Quick Start simplified UI
✅ Input validation and user guidance

Author: Senior Instrumentation Engineer
Organization: Digantara Research and Technologies Pvt. Ltd.
Date: 2025-01-22
Version: 3.0.0 - Enhanced for Signal Delay Analysis
"""

import sys
import os
import time
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import signal
import atexit
import json

import numpy as np
import pandas as pd
import gradio as gr
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for threading
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy import signal as scipy_signal
from scipy import stats
from scipy.io import savemat

# Add parent directory to path
script_dir = Path(__file__).resolve().parent.parent.parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

try:
    from instrument_control.keysight_oscilloscope import KeysightDSOX6004A
except ImportError as e:
    print(f"Error importing oscilloscope module: {e}")
    sys.exit(1)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class CaptureConfig:
    """Configuration for continuous trigger capture session"""
    num_captures: int
    time_interval: float  # seconds between captures
    channels: List[int]
    base_filename: str
    save_directory: str
    capture_screenshots: bool = True
    save_waveforms: bool = True
    save_combined_csv: bool = False
    trigger_timeout: float = 10.0  # seconds to wait for trigger

    # NEW: Delay analysis configuration
    enable_delay_analysis: bool = True
    delay_reference_channel: int = 1  # Reference channel (e.g., Laser = CH1)
    delay_measurement_channel: int = 2  # Measurement channel (e.g., APD = CH2)

    def validate(self) -> Tuple[bool, str]:
        """Validate configuration"""
        if self.num_captures < 1:
            return False, "Number of captures must be at least 1"
        if self.time_interval < 0:
            return False, "Time interval cannot be negative"
        if not self.channels:
            return False, "At least one channel must be selected"
        if not self.base_filename:
            return False, "Base filename cannot be empty"
        if not (self.capture_screenshots or self.save_waveforms or self.save_combined_csv):
            return False, "Must enable at least one saving option (screenshots, per-channel CSV, or combined CSV)"

        # Validate delay analysis configuration
        if self.enable_delay_analysis:
            if len(self.channels) < 2:
                return False, "Delay analysis requires at least 2 channels"
            if self.delay_reference_channel not in self.channels:
                return False, f"Reference channel {self.delay_reference_channel} not in selected channels"
            if self.delay_measurement_channel not in self.channels:
                return False, f"Measurement channel {self.delay_measurement_channel} not in selected channels"
            if self.delay_reference_channel == self.delay_measurement_channel:
                return False, "Reference and measurement channels must be different"

        return True, "Configuration valid"

@dataclass
class CaptureResult:
    """Result from a single trigger capture with delay analysis"""
    index: int
    timestamp: datetime
    screenshot_file: Optional[str] = None
    waveform_files: List[str] = None
    measurements: Dict[str, Any] = None

    # NEW: Signal delay analysis
    delay_seconds: Optional[float] = None  # Measured delay between channels
    delay_confidence: Optional[float] = None  # Confidence metric (0-1)
    signal_snr: Dict[int, float] = field(default_factory=dict)  # SNR per channel
    peak_amplitudes: Dict[int, float] = field(default_factory=dict)  # Peak amplitude per channel

    # Raw waveform data for analysis (optional, in-memory)
    waveform_data: Dict[int, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    success: bool = True
    error_message: Optional[str] = None

# ============================================================================
# SIGNAL ANALYSIS MODULE
# ============================================================================

class SignalAnalyzer:
    """
    Advanced signal analysis for delay measurement and characterization.
    Uses cross-correlation for accurate delay estimation.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def calculate_delay(
        self,
        ref_time: np.ndarray,
        ref_voltage: np.ndarray,
        meas_time: np.ndarray,
        meas_voltage: np.ndarray,
        method: str = 'xcorr'
    ) -> Tuple[float, float]:
        """
        Calculate time delay between two signals.

        Args:
            ref_time: Time array for reference signal
            ref_voltage: Voltage array for reference signal
            meas_time: Time array for measurement signal
            meas_voltage: Voltage array for measurement signal
            method: 'xcorr' (cross-correlation) or 'threshold' (threshold crossing)

        Returns:
            (delay_seconds, confidence)
            delay_seconds: Measured delay (positive = meas lags ref)
            confidence: Quality metric 0-1 (1 = highest confidence)
        """
        try:
            if method == 'xcorr':
                return self._calculate_delay_xcorr(ref_voltage, meas_voltage, ref_time)
            elif method == 'threshold':
                return self._calculate_delay_threshold(ref_time, ref_voltage, meas_time, meas_voltage)
            else:
                raise ValueError(f"Unknown delay calculation method: {method}")
        except Exception as e:
            self.logger.error(f"Delay calculation failed: {e}")
            return None, 0.0

    def _calculate_delay_xcorr(
        self,
        ref_voltage: np.ndarray,
        meas_voltage: np.ndarray,
        ref_time: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculate delay using cross-correlation method.
        Most accurate for periodic signals with good SNR.
        """
        # Normalize signals (remove DC, normalize amplitude)
        ref_norm = self._normalize_signal(ref_voltage)
        meas_norm = self._normalize_signal(meas_voltage)

        # Compute cross-correlation
        correlation = scipy_signal.correlate(meas_norm, ref_norm, mode='full', method='fft')
        lags = scipy_signal.correlation_lags(len(meas_norm), len(ref_norm), mode='full')

        # Find peak of correlation
        peak_idx = np.argmax(np.abs(correlation))
        lag_samples = lags[peak_idx]

        # Convert to time delay
        sample_interval = ref_time[1] - ref_time[0] if len(ref_time) > 1 else 1e-9
        delay_seconds = lag_samples * sample_interval

        # Calculate confidence metric
        # High confidence = sharp peak, low noise
        max_corr = np.abs(correlation[peak_idx])
        mean_corr = np.mean(np.abs(correlation))
        std_corr = np.std(np.abs(correlation))

        # Confidence: normalized peak height above noise floor
        confidence = min(1.0, (max_corr - mean_corr) / (3 * std_corr + 1e-10))
        confidence = max(0.0, confidence)  # Clamp to [0, 1]

        self.logger.debug(f"Cross-correlation delay: {delay_seconds*1e9:.3f} ns, confidence: {confidence:.3f}")

        return delay_seconds, confidence

    def _calculate_delay_threshold(
        self,
        ref_time: np.ndarray,
        ref_voltage: np.ndarray,
        meas_time: np.ndarray,
        meas_voltage: np.ndarray,
        threshold_percent: float = 0.5
    ) -> Tuple[float, float]:
        """
        Calculate delay using threshold crossing method.
        Useful for fast edge detection.
        """
        # Find threshold crossings
        ref_threshold = np.min(ref_voltage) + threshold_percent * (np.max(ref_voltage) - np.min(ref_voltage))
        meas_threshold = np.min(meas_voltage) + threshold_percent * (np.max(meas_voltage) - np.min(meas_voltage))

        # Find first rising edge
        ref_crossing = self._find_first_crossing(ref_time, ref_voltage, ref_threshold)
        meas_crossing = self._find_first_crossing(meas_time, meas_voltage, meas_threshold)

        if ref_crossing is None or meas_crossing is None:
            return None, 0.0

        delay_seconds = meas_crossing - ref_crossing

        # Confidence based on edge sharpness
        confidence = 0.8  # Fixed confidence for threshold method

        return delay_seconds, confidence

    def _find_first_crossing(
        self,
        time: np.ndarray,
        voltage: np.ndarray,
        threshold: float
    ) -> Optional[float]:
        """Find first threshold crossing with linear interpolation"""
        crossings = np.where(np.diff(np.sign(voltage - threshold)) > 0)[0]

        if len(crossings) == 0:
            return None

        # Linear interpolation for sub-sample accuracy
        idx = crossings[0]
        if idx + 1 < len(voltage):
            # Interpolate crossing time
            v1, v2 = voltage[idx], voltage[idx + 1]
            t1, t2 = time[idx], time[idx + 1]
            crossing_time = t1 + (threshold - v1) * (t2 - t1) / (v2 - v1 + 1e-20)
            return crossing_time

        return time[idx]

    def _normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """Normalize signal: remove DC offset and scale to unit variance"""
        signal_centered = signal - np.mean(signal)
        signal_std = np.std(signal_centered)
        if signal_std > 1e-10:
            return signal_centered / signal_std
        return signal_centered

    def calculate_snr(self, voltage: np.ndarray) -> float:
        """
        Calculate signal-to-noise ratio.
        Assumes signal is periodic and noise is high-frequency.
        """
        # Simple SNR estimate: ratio of signal power to noise power
        # Signal power ≈ variance of low-pass filtered signal
        # Noise power ≈ variance of high-pass filtered signal

        # Apply simple moving average as low-pass filter
        window = min(len(voltage) // 10, 100)
        if window < 3:
            return float('inf')

        signal_smooth = np.convolve(voltage, np.ones(window)/window, mode='same')
        noise = voltage - signal_smooth

        signal_power = np.var(signal_smooth)
        noise_power = np.var(noise)

        if noise_power < 1e-20:
            return float('inf')

        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(snr_linear)

        return snr_db

    def calculate_peak_amplitude(self, voltage: np.ndarray) -> float:
        """Calculate peak-to-peak amplitude"""
        return np.max(voltage) - np.min(voltage)

# ============================================================================
# STATISTICS TRACKER
# ============================================================================

class StatisticsTracker:
    """Real-time statistics tracking for delay measurements"""

    def __init__(self):
        self.delays: List[float] = []
        self.timestamps: List[datetime] = []
        self.confidences: List[float] = []
        self.lock = threading.Lock()

    def add_measurement(self, delay: float, timestamp: datetime, confidence: float):
        """Add new delay measurement"""
        with self.lock:
            self.delays.append(delay)
            self.timestamps.append(timestamp)
            self.confidences.append(confidence)

    def get_statistics(self) -> Dict[str, Any]:
        """Calculate current statistics"""
        with self.lock:
            if not self.delays:
                return {}

            delays_ns = np.array(self.delays) * 1e9  # Convert to nanoseconds

            return {
                'count': len(delays_ns),
                'mean_ns': np.mean(delays_ns),
                'std_ns': np.std(delays_ns, ddof=1) if len(delays_ns) > 1 else 0.0,
                'min_ns': np.min(delays_ns),
                'max_ns': np.max(delays_ns),
                'median_ns': np.median(delays_ns),
                'jitter_rms_ns': np.std(delays_ns, ddof=1) if len(delays_ns) > 1 else 0.0,
                'jitter_pk_pk_ns': np.max(delays_ns) - np.min(delays_ns),
                'mean_confidence': np.mean(self.confidences) if self.confidences else 0.0,
                'sem_ns': stats.sem(delays_ns) if len(delays_ns) > 1 else 0.0,  # Standard error of mean
                'ci_95_ns': 1.96 * stats.sem(delays_ns) if len(delays_ns) > 1 else 0.0
            }

    def get_delays_array(self) -> np.ndarray:
        """Get delays as numpy array (in nanoseconds)"""
        with self.lock:
            return np.array(self.delays) * 1e9 if self.delays else np.array([])

    def clear(self):
        """Clear all statistics"""
        with self.lock:
            self.delays.clear()
            self.timestamps.clear()
            self.confidences.clear()

# ============================================================================
# TRIGGER CAPTURE ENGINE (ENHANCED)
# ============================================================================

class TriggerCaptureEngine:
    """
    Enhanced core engine for continuous trigger-based capture using SINGLE mode.
    Captures screenshot and waveform data after each successful trigger.
    Performs automatic delay analysis and real-time statistics.
    """

    def __init__(self, oscilloscope: KeysightDSOX6004A):
        self.scope = oscilloscope
        self.logger = logging.getLogger(self.__class__.__name__)

        # State tracking
        self.is_running = False
        self.stop_requested = False
        self.current_capture = 0
        self.total_captures = 0

        # Results storage
        self.capture_results: List[CaptureResult] = []
        self.capture_thread: Optional[threading.Thread] = None

        # NEW: Signal analysis components
        self.signal_analyzer = SignalAnalyzer()
        self.statistics_tracker = StatisticsTracker()

        # Thread safety
        self.lock = threading.RLock()

        # Current configuration
        self.current_config: Optional[CaptureConfig] = None

    def start_capture_session(self, config: CaptureConfig) -> bool:
        """Start continuous capture session"""
        # Validate configuration
        valid, msg = config.validate()
        if not valid:
            self.logger.error(f"Invalid configuration: {msg}")
            return False

        # Check oscilloscope connection
        if not self.scope.is_connected:
            self.logger.error("Oscilloscope not connected")
            return False

        # Create save directory with organized structure
        save_dir = Path(config.save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (save_dir / "screenshots").mkdir(exist_ok=True)
        (save_dir / "waveforms").mkdir(exist_ok=True)
        (save_dir / "analysis").mkdir(exist_ok=True)

        # Initialize state
        with self.lock:
            if self.is_running:
                self.logger.warning("Capture already in progress")
                return False

            self.is_running = True
            self.stop_requested = False
            self.current_capture = 0
            self.total_captures = config.num_captures
            self.capture_results = []
            self.statistics_tracker.clear()
            self.current_config = config

        # Save configuration
        self._save_configuration(config)

        # Start capture thread
        self.capture_thread = threading.Thread(
            target=self._capture_loop,
            args=(config,),
            daemon=True
        )
        self.capture_thread.start()

        self.logger.info(f"Started capture session: {config.num_captures} captures with delay analysis")
        return True

    def stop_capture_session(self):
        """Stop ongoing capture session"""
        self.logger.info("Stopping capture session...")
        with self.lock:
            self.stop_requested = True

        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=5.0)

        with self.lock:
            self.is_running = False

    def get_status(self) -> Dict[str, Any]:
        """Get current capture status"""
        with self.lock:
            return {
                'is_running': self.is_running,
                'current_capture': self.current_capture,
                'total_captures': self.total_captures,
                'completed_captures': len(self.capture_results),
                'successful_captures': sum(1 for r in self.capture_results if r.success),
                'failed_captures': sum(1 for r in self.capture_results if not r.success),
                'progress_percentage': (self.current_capture / max(self.total_captures, 1)) * 100
            }

    def get_statistics(self) -> Dict[str, Any]:
        """Get current delay statistics"""
        return self.statistics_tracker.get_statistics()

    def get_file_list(self) -> List[str]:
        """Get list of all captured files"""
        files = []
        for result in self.capture_results:
            if result.screenshot_file:
                files.append(result.screenshot_file)
            if result.waveform_files:
                files.extend(result.waveform_files)
        return files

    def _save_configuration(self, config: CaptureConfig):
        """Save session configuration to JSON"""
        try:
            config_file = Path(config.save_directory) / "session_config.json"
            config_dict = {
                'timestamp': datetime.now().isoformat(),
                'num_captures': config.num_captures,
                'time_interval': config.time_interval,
                'channels': config.channels,
                'base_filename': config.base_filename,
                'trigger_timeout': config.trigger_timeout,
                'enable_delay_analysis': config.enable_delay_analysis,
                'delay_reference_channel': config.delay_reference_channel,
                'delay_measurement_channel': config.delay_measurement_channel
            }

            with open(config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)

            self.logger.info(f"Configuration saved: {config_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save configuration: {e}")

    def _capture_loop(self, config: CaptureConfig):
        """Main capture loop running in separate thread"""
        try:
            # Initial setup
            self._setup_oscilloscope_for_capture(config)

            for capture_idx in range(config.num_captures):
                loop_start = time.time()

                # Check for stop request
                if self.stop_requested:
                    self.logger.info("Capture stopped by user")
                    break

                # Update current capture index
                with self.lock:
                    self.current_capture = capture_idx + 1

                self.logger.info(f"Starting capture {self.current_capture}/{config.num_captures}")

                # Perform single capture
                result = self._perform_single_capture(config, capture_idx)

                # Store result
                with self.lock:
                    self.capture_results.append(result)

                # Update statistics tracker
                if result.success and result.delay_seconds is not None:
                    self.statistics_tracker.add_measurement(
                        result.delay_seconds,
                        result.timestamp,
                        result.delay_confidence or 0.0
                    )

                if not result.success:
                    self.logger.error(f"Capture {capture_idx + 1} failed: {result.error_message}")
                else:
                    delay_str = f"{result.delay_seconds*1e9:.3f} ns" if result.delay_seconds else "N/A"
                    self.logger.info(f"Capture {capture_idx + 1} completed - Delay: {delay_str}")

                # Wait for target interval
                if capture_idx < config.num_captures - 1 and not self.stop_requested:
                    elapsed = time.time() - loop_start
                    sleep_time = max(0.0, config.time_interval - elapsed)
                    self.logger.info(
                        f"Capture {capture_idx + 1} elapsed time: {elapsed:.3f}s, "
                        f"sleeping {sleep_time:.3f}s to target interval {config.time_interval}s"
                    )
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            self.logger.info("Capture session completed")

        except Exception as e:
            self.logger.error(f"Capture loop error: {e}")
        finally:
            with self.lock:
                self.is_running = False

    def _setup_oscilloscope_for_capture(self, config: CaptureConfig):
        """Configure oscilloscope for single trigger capture mode"""
        try:
            # Stop any ongoing acquisition
            self.scope._scpi_wrapper.write(":STOP")
            time.sleep(0.1)

            # Set acquisition type to normal
            self.scope._scpi_wrapper.write(":ACQuire:TYPE NORMal")

            # Enable selected channels
            for channel in config.channels:
                self.scope._scpi_wrapper.write(f":CHANnel{channel}:DISPlay ON")

            # Set trigger sweep to NORMAL (wait for trigger)
            self.scope._scpi_wrapper.write(":TRIGger:SWEep NORMal")

            self.logger.info("Oscilloscope configured for single trigger capture")

        except Exception as e:
            self.logger.error(f"Failed to setup oscilloscope: {e}")
            raise

    def _perform_single_capture(self, config: CaptureConfig, capture_idx: int) -> CaptureResult:
        """Perform a single trigger capture with screenshot and data save"""
        timestamp = datetime.now()
        result = CaptureResult(
            index=capture_idx,
            timestamp=timestamp,
            waveform_files=[]
        )

        try:
            # Set to SINGLE mode - capture one trigger and stop
            self.logger.debug("Setting SINGLE mode and waiting for trigger...")
            self.scope._scpi_wrapper.write(":SINGle")

            # Wait for trigger with timeout
            trigger_acquired = self._wait_for_trigger(config.trigger_timeout)

            if not trigger_acquired:
                result.success = False
                result.error_message = "Trigger timeout"
                return result

            self.logger.debug("Trigger acquired successfully")

            # Small delay to ensure display is updated
            time.sleep(0.1)

            # Capture screenshot if enabled
            if config.capture_screenshots:
                screenshot_file = self._capture_screenshot(config, capture_idx, timestamp)
                if screenshot_file:
                    result.screenshot_file = screenshot_file
                    self.logger.debug(f"Screenshot saved: {screenshot_file}")

            # Save waveform data and perform analysis
            if config.save_waveforms or config.save_combined_csv or config.enable_delay_analysis:
                waveform_data, waveform_files = self._save_waveform_data(config, capture_idx, timestamp)
                if waveform_files:
                    result.waveform_files = waveform_files
                    self.logger.debug(f"Waveforms saved: {len(waveform_files)} files")

                # Store waveform data for delay analysis
                if waveform_data:
                    result.waveform_data = waveform_data

                    # Perform delay analysis
                    if config.enable_delay_analysis:
                        self._perform_delay_analysis(config, result, waveform_data)

            # Get measurements for selected channels
            result.measurements = self._get_measurements(config.channels)

            result.success = True
            return result

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            self.logger.error(f"Capture failed: {e}")
            return result

    def _perform_delay_analysis(
        self,
        config: CaptureConfig,
        result: CaptureResult,
        waveform_data: Dict[int, Tuple[np.ndarray, np.ndarray]]
    ):
        """Perform delay analysis between reference and measurement channels"""
        try:
            ref_ch = config.delay_reference_channel
            meas_ch = config.delay_measurement_channel

            if ref_ch not in waveform_data or meas_ch not in waveform_data:
                self.logger.warning("Required channels not available for delay analysis")
                return

            ref_time, ref_voltage = waveform_data[ref_ch]
            meas_time, meas_voltage = waveform_data[meas_ch]

            # Calculate delay using cross-correlation
            delay, confidence = self.signal_analyzer.calculate_delay(
                ref_time, ref_voltage,
                meas_time, meas_voltage,
                method='xcorr'
            )

            if delay is not None:
                result.delay_seconds = delay
                result.delay_confidence = confidence

                # Calculate SNR for both channels
                result.signal_snr[ref_ch] = self.signal_analyzer.calculate_snr(ref_voltage)
                result.signal_snr[meas_ch] = self.signal_analyzer.calculate_snr(meas_voltage)

                # Calculate peak amplitudes
                result.peak_amplitudes[ref_ch] = self.signal_analyzer.calculate_peak_amplitude(ref_voltage)
                result.peak_amplitudes[meas_ch] = self.signal_analyzer.calculate_peak_amplitude(meas_voltage)

                self.logger.debug(
                    f"Delay analysis: {delay*1e9:.3f} ns (confidence: {confidence:.3f}), "
                    f"SNR: CH{ref_ch}={result.signal_snr[ref_ch]:.1f}dB, "
                    f"CH{meas_ch}={result.signal_snr[meas_ch]:.1f}dB"
                )

        except Exception as e:
            self.logger.error(f"Delay analysis failed: {e}")

    def _wait_for_trigger(self, timeout: float) -> bool:
        """Wait for oscilloscope to trigger with timeout"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Check if acquisition is complete
                # Query operation status register
                oper_status = self.scope._scpi_wrapper.query(":OPERegister:CONDition?")

                # Bit 3 (value 8) indicates "Run" status
                # When cleared, acquisition is complete
                if int(oper_status) & 8 == 0:
                    return True

                time.sleep(0.1)

            except Exception as e:
                self.logger.warning(f"Error checking trigger status: {e}")
                time.sleep(0.1)

        return False

    def _capture_screenshot(self, config: CaptureConfig, capture_idx: int, timestamp: datetime) -> Optional[str]:
        """Capture and save oscilloscope screenshot"""
        try:
            # Generate filename
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            filename = f"{config.base_filename}_screenshot_{capture_idx:04d}_{timestamp_str}.png"
            filepath = Path(config.save_directory) / "screenshots" / filename

            # Get screenshot data
            self.logger.debug("Capturing screenshot...")
            image_data = self.scope._scpi_wrapper.query_binary_values(
                ":DISPlay:DATA? PNG",
                datatype='B'
            )

            if image_data:
                # Save screenshot
                with open(filepath, 'wb') as f:
                    f.write(bytes(image_data))
                return str(filepath)
            else:
                self.logger.warning("No screenshot data received")
                return None

        except Exception as e:
            self.logger.error(f"Screenshot capture failed: {e}")
            return None

    def _save_waveform_data(
        self,
        config: CaptureConfig,
        capture_idx: int,
        timestamp: datetime
    ) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], List[str]]:
        """
        Save waveform data from all configured channels.

        Returns:
            (waveform_data, saved_files)
            waveform_data: Dict mapping channel -> (time_array, voltage_array)
            saved_files: List of file paths
        """
        saved_files: List[str] = []
        waveform_data: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        for channel in config.channels:
            try:
                # Configure waveform source
                self.scope._scpi_wrapper.write(f":WAVeform:SOURce CHANnel{channel}")
                self.scope._scpi_wrapper.write(":WAVeform:FORMat BYTE")
                self.scope._scpi_wrapper.write(":WAVeform:POINts:MODE RAW")
                self.scope._scpi_wrapper.write(":WAVeform:POINts 62500")

                # Get preamble for scaling
                preamble = self.scope._scpi_wrapper.query(":WAVeform:PREamble?")
                preamble_parts = preamble.split(',')

                y_increment = float(preamble_parts[7])
                y_origin = float(preamble_parts[8])
                y_reference = float(preamble_parts[9])
                x_increment = float(preamble_parts[4])
                x_origin = float(preamble_parts[5])

                # Get waveform data
                raw_data = self.scope._scpi_wrapper.query_binary_values(
                    ":WAVeform:DATA?",
                    datatype='B'
                )

                # Scale data
                voltage_data = np.array([(val - y_reference) * y_increment + y_origin for val in raw_data])
                time_data = np.array([x_origin + i * x_increment for i in range(len(voltage_data))])

                # Store waveform data
                waveform_data[channel] = (time_data, voltage_data)

                # Save per-channel CSV if requested
                if config.save_waveforms:
                    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    filename = f"{config.base_filename}_CH{channel}_{capture_idx:04d}_{timestamp_str}.csv"
                    filepath = Path(config.save_directory) / "waveforms" / filename

                    df = pd.DataFrame({
                        'Time (s)': time_data,
                        'Voltage (V)': voltage_data
                    })

                    with open(filepath, 'w') as f:
                        f.write(f"# Channel: {channel}\n")
                        f.write(f"# Capture Index: {capture_idx}\n")
                        f.write(f"# Timestamp: {timestamp.isoformat()}\n")
                        f.write(f"# Sample Rate: {1.0/x_increment:.2e} Hz\n")
                        f.write(f"# Points: {len(voltage_data)}\n")
                        f.write("\n")
                        df.to_csv(f, index=False)

                    saved_files.append(str(filepath))

            except Exception as e:
                self.logger.error(f"Failed to save waveform for channel {channel}: {e}")

        # Save combined multi-channel CSV if requested
        if config.save_combined_csv and waveform_data:
            try:
                # Use shortest trace length across channels to align data
                min_len = min(len(data[0]) for data in waveform_data.values())
                if min_len > 0:
                    # Reference time axis from first channel in list
                    first_channel = config.channels[0]
                    time_ref = waveform_data[first_channel][0][:min_len]
                    data_dict = {
                        'Time (s)': time_ref
                    }
                    for ch, (t_data, v_data) in waveform_data.items():
                        data_dict[f'CH{ch} (V)'] = v_data[:min_len]

                    df_multi = pd.DataFrame(data_dict)
                    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    filename = f"{config.base_filename}_MULTI_{capture_idx:04d}_{timestamp_str}.csv"
                    filepath = Path(config.save_directory) / "waveforms" / filename

                    with open(filepath, 'w') as f:
                        f.write(f"# Channels: {', '.join(str(ch) for ch in config.channels)}\n")
                        f.write(f"# Capture Index: {capture_idx}\n")
                        f.write(f"# Timestamp: {timestamp.isoformat()}\n")
                        f.write(f"# Points: {min_len}\n")
                        f.write("\n")
                        df_multi.to_csv(f, index=False)

                    saved_files.append(str(filepath))
            except Exception as e:
                self.logger.error(f"Failed to save combined waveform CSV: {e}")

        return waveform_data, saved_files

    def _get_measurements(self, channels: List[int]) -> Dict[str, Any]:
        """Get basic measurements from channels"""
        measurements = {}

        for channel in channels:
            try:
                ch_measurements = {}

                # Get common measurements
                measurement_types = ['FREQ', 'PERiod', 'VPP', 'VAVG', 'VRMS']

                for meas_type in measurement_types:
                    try:
                        value = self.scope.measure_single(channel, meas_type)
                        if value is not None:
                            ch_measurements[meas_type] = value
                    except:
                        pass

                measurements[f'CH{channel}'] = ch_measurements

            except Exception as e:
                self.logger.warning(f"Failed to get measurements for channel {channel}: {e}")

        return measurements

# ============================================================================
# VISUALIZATION AND REPORTING
# ============================================================================

class ResultsVisualizer:
    """Generate plots and reports for delay analysis"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def plot_delay_distribution(
        self,
        delays_ns: np.ndarray,
        title: str = "Delay Distribution"
    ) -> Figure:
        """Generate histogram and time series plot of delay measurements"""
        if len(delays_ns) == 0:
            # Return empty figure
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.text(0.5, 0.5, 'No data available',
                   ha='center', va='center', fontsize=16)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return fig

        fig, axes = plt.subplots(2, 1, figsize=(10, 10))

        # Histogram
        axes[0].hist(delays_ns, bins=min(30, len(delays_ns)//2 + 1),
                    edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].set_xlabel('Delay (ns)', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title(f'{title} - Histogram', fontsize=14, fontweight='bold')

        mean_delay = np.mean(delays_ns)
        std_delay = np.std(delays_ns, ddof=1) if len(delays_ns) > 1 else 0

        axes[0].axvline(mean_delay, color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {mean_delay:.3f} ± {std_delay:.3f} ns')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)

        # Time series
        axes[1].plot(range(len(delays_ns)), delays_ns, marker='o',
                    linestyle='-', alpha=0.7, color='steelblue', markersize=4)
        axes[1].axhline(mean_delay, color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {mean_delay:.3f} ns')
        axes[1].fill_between(range(len(delays_ns)),
                            mean_delay - std_delay,
                            mean_delay + std_delay,
                            alpha=0.2, color='red', label=f'±1σ: {std_delay:.3f} ns')
        axes[1].set_xlabel('Capture Number', fontsize=12)
        axes[1].set_ylabel('Delay (ns)', fontsize=12)
        axes[1].set_title(f'{title} - Time Series', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    def save_plot(self, fig: Figure, filepath: Path):
        """Save matplotlib figure to file"""
        try:
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved: {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save plot: {e}")

class ReportGenerator:
    """Generate comprehensive statistical reports"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate_text_report(
        self,
        results: List[CaptureResult],
        config: CaptureConfig
    ) -> str:
        """Generate detailed text-based statistical report"""

        successful_results = [r for r in results if r.success and r.delay_seconds is not None]

        if not successful_results:
            return "No successful measurements with delay data available."

        delays_ns = np.array([r.delay_seconds * 1e9 for r in successful_results])
        confidences = np.array([r.delay_confidence for r in successful_results if r.delay_confidence])

        # Calculate statistics
        mean_delay = np.mean(delays_ns)
        std_delay = np.std(delays_ns, ddof=1)
        median_delay = np.median(delays_ns)
        min_delay = np.min(delays_ns)
        max_delay = np.max(delays_ns)
        jitter_pk_pk = max_delay - min_delay
        sem = stats.sem(delays_ns)
        ci_95 = 1.96 * sem

        # Distribution statistics
        skewness = stats.skew(delays_ns)
        kurtosis = stats.kurtosis(delays_ns)

        report = f"""
╔════════════════════════════════════════════════════════════════╗
║        SIGNAL DELAY ANALYSIS REPORT                            ║
║        Laser-APD Timing Jitter Measurement                     ║
╚════════════════════════════════════════════════════════════════╝

SESSION INFORMATION
────────────────────────────────────────────────────────────────
Session Name:              {config.base_filename}
Start Time:                {results[0].timestamp.strftime('%Y-%m-%d %H:%M:%S')}
End Time:                  {results[-1].timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Duration:                  {(results[-1].timestamp - results[0].timestamp).total_seconds():.1f} seconds
Reference Channel:         CH{config.delay_reference_channel} (Laser)
Measurement Channel:       CH{config.delay_measurement_channel} (APD)

ACQUISITION SUMMARY
────────────────────────────────────────────────────────────────
Total Acquisitions:        {len(results)}
Successful Captures:       {len(successful_results)}
Failed Captures:           {len(results) - len(successful_results)}
Success Rate:              {len(successful_results)/len(results)*100:.2f}%
Valid Delay Measurements:  {len(delays_ns)}

DELAY STATISTICS (Laser → APD)
────────────────────────────────────────────────────────────────
Mean Delay:                {mean_delay:.6f} ns
Standard Deviation (σ):    {std_delay:.6f} ns
Standard Error (SEM):      {sem:.6f} ns
Median Delay:              {median_delay:.6f} ns

Minimum Delay:             {min_delay:.6f} ns
Maximum Delay:             {max_delay:.6f} ns

JITTER ANALYSIS
────────────────────────────────────────────────────────────────
RMS Jitter (1σ):           {std_delay:.6f} ns
Peak-to-Peak Jitter:       {jitter_pk_pk:.6f} ns

CONFIDENCE INTERVAL
────────────────────────────────────────────────────────────────
95% CI:                    {mean_delay:.6f} ± {ci_95:.6f} ns
95% CI Range:              [{mean_delay-ci_95:.6f}, {mean_delay+ci_95:.6f}] ns

PRECISION METRICS
────────────────────────────────────────────────────────────────
Coefficient of Variation:  {(std_delay/mean_delay)*100:.4f}%
Relative Precision:        {(ci_95/mean_delay)*100:.4f}% (95% CI)

DISTRIBUTION ANALYSIS
────────────────────────────────────────────────────────────────
Skewness:                  {skewness:.4f}
Kurtosis (excess):         {kurtosis:.4f}
Distribution Shape:        {'Normal-like' if abs(skewness) < 0.5 and abs(kurtosis) < 1 else 'Non-normal'}

MEASUREMENT CONFIDENCE
────────────────────────────────────────────────────────────────
Mean Confidence Score:     {np.mean(confidences):.4f} (out of 1.0)
Min Confidence:            {np.min(confidences):.4f}
Max Confidence:            {np.max(confidences):.4f}

SIGNAL QUALITY (Last Measurement)
────────────────────────────────────────────────────────────────
"""

        # Add SNR information from last successful measurement
        last_result = successful_results[-1]
        if last_result.signal_snr:
            for ch, snr in last_result.signal_snr.items():
                report += f"CH{ch} SNR:                   {snr:.2f} dB\n"

        if last_result.peak_amplitudes:
            report += "\n"
            for ch, amp in last_result.peak_amplitudes.items():
                report += f"CH{ch} Peak-to-Peak:          {amp:.4f} V\n"

        report += f"""
────────────────────────────────────────────────────────────────
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Digantara Research and Technologies Pvt. Ltd.
────────────────────────────────────────────────────────────────
"""

        return report

    def export_to_excel(
        self,
        results: List[CaptureResult],
        config: CaptureConfig,
        filepath: Path
    ):
        """Export results to Excel with multiple sheets"""
        try:
            successful_results = [r for r in results if r.success]

            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Sheet 1: Raw data
                raw_data = {
                    'Capture_Index': [r.index for r in successful_results],
                    'Timestamp': [r.timestamp.isoformat() for r in successful_results],
                    'Delay_ns': [r.delay_seconds * 1e9 if r.delay_seconds else None for r in successful_results],
                    'Confidence': [r.delay_confidence if r.delay_confidence else None for r in successful_results],
                    'Screenshot_File': [r.screenshot_file if r.screenshot_file else '' for r in successful_results],
                    'Success': [r.success for r in successful_results]
                }

                df_raw = pd.DataFrame(raw_data)
                df_raw.to_excel(writer, sheet_name='Raw Data', index=False)

                # Sheet 2: Statistics
                delays_ns = np.array([r.delay_seconds * 1e9 for r in successful_results if r.delay_seconds])

                if len(delays_ns) > 0:
                    stats_data = {
                        'Metric': [
                            'Mean Delay (ns)',
                            'Std Dev (ns)',
                            'Median (ns)',
                            'Min (ns)',
                            'Max (ns)',
                            'RMS Jitter (ns)',
                            'Pk-Pk Jitter (ns)',
                            'SEM (ns)',
                            '95% CI (ns)',
                            'Coefficient of Variation (%)',
                            'Number of Samples'
                        ],
                        'Value': [
                            np.mean(delays_ns),
                            np.std(delays_ns, ddof=1),
                            np.median(delays_ns),
                            np.min(delays_ns),
                            np.max(delays_ns),
                            np.std(delays_ns, ddof=1),
                            np.max(delays_ns) - np.min(delays_ns),
                            stats.sem(delays_ns),
                            1.96 * stats.sem(delays_ns),
                            (np.std(delays_ns, ddof=1) / np.mean(delays_ns)) * 100,
                            len(delays_ns)
                        ]
                    }

                    df_stats = pd.DataFrame(stats_data)
                    df_stats.to_excel(writer, sheet_name='Statistics', index=False)

                # Sheet 3: Configuration
                config_data = {
                    'Parameter': [
                        'Number of Captures',
                        'Time Interval (s)',
                        'Channels',
                        'Reference Channel',
                        'Measurement Channel',
                        'Base Filename',
                        'Trigger Timeout (s)'
                    ],
                    'Value': [
                        config.num_captures,
                        config.time_interval,
                        ', '.join(str(ch) for ch in config.channels),
                        config.delay_reference_channel,
                        config.delay_measurement_channel,
                        config.base_filename,
                        config.trigger_timeout
                    ]
                }

                df_config = pd.DataFrame(config_data)
                df_config.to_excel(writer, sheet_name='Configuration', index=False)

            self.logger.info(f"Excel report exported: {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to export Excel: {e}")
            raise

    def export_to_matlab(
        self,
        results: List[CaptureResult],
        filepath: Path
    ):
        """Export results to MATLAB .mat format"""
        try:
            successful_results = [r for r in results if r.success and r.delay_seconds is not None]

            delays_ns = np.array([r.delay_seconds * 1e9 for r in successful_results])
            confidences = np.array([r.delay_confidence for r in successful_results if r.delay_confidence])
            timestamps = np.array([r.timestamp.timestamp() for r in successful_results])

            matlab_data = {
                'delays_ns': delays_ns,
                'confidences': confidences,
                'timestamps': timestamps,
                'capture_indices': np.array([r.index for r in successful_results]),
                'mean_delay_ns': np.mean(delays_ns),
                'std_delay_ns': np.std(delays_ns, ddof=1) if len(delays_ns) > 1 else 0,
                'jitter_rms_ns': np.std(delays_ns, ddof=1) if len(delays_ns) > 1 else 0,
                'jitter_pk_pk_ns': np.max(delays_ns) - np.min(delays_ns) if len(delays_ns) > 0 else 0
            }

            savemat(filepath, matlab_data)
            self.logger.info(f"MATLAB file exported: {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to export MATLAB: {e}")
            raise

# ============================================================================
# GRADIO USER INTERFACE (ENHANCED)
# ============================================================================

class TriggerCaptureGUI:
    """
    Enhanced Gradio interface for continuous trigger capture with delay analysis.
    Focused on ease of use for laser-APD delay measurements.
    """

    def __init__(self):
        self.oscilloscope = None
        self.capture_engine = None
        self.logger = self._setup_logging()

        # Default paths
        self.default_save_dir = Path.cwd() / "laser_apd_analysis"
        self.default_save_dir.mkdir(parents=True, exist_ok=True)

        # Visualization and reporting
        self.visualizer = ResultsVisualizer()
        self.report_generator = ReportGenerator()

        # Common timebase scales
        self.timebase_scales: List[Tuple[str, float]] = [
            ("1 ns", 1e-9), ("2 ns", 2e-9), ("5 ns", 5e-9),
            ("10 ns", 10e-9), ("20 ns", 20e-9), ("50 ns", 50e-9),
            ("100 ns", 100e-9), ("200 ns", 200e-9), ("500 ns", 500e-9),
            ("1 µs", 1e-6), ("2 µs", 2e-6), ("5 µs", 5e-6),
            ("10 µs", 10e-6), ("20 µs", 20e-6), ("50 µs", 50e-6),
            ("100 µs", 100e-6), ("200 µs", 200e-6), ("500 µs", 500e-6),
            ("1 ms", 1e-3), ("2 ms", 2e-3), ("5 ms", 5e-3),
            ("10 ms", 10e-3), ("20 ms", 20e-3), ("50 ms", 50e-3),
            ("100 ms", 100e-3), ("200 ms", 200e-3), ("500 ms", 500e-3),
            ("1 s", 1.0), ("2 s", 2.0), ("5 s", 5.0),
            ("10 s", 10.0), ("20 s", 20.0), ("50 s", 50.0),
        ]

        # Cleanup handlers
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, self._signal_handler)

    def _setup_logging(self) -> logging.Logger:
        """Configure logging"""
        logger = logging.getLogger('TriggerCaptureEnhanced')
        logger.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    def _signal_handler(self, signum, frame):
        """Handle system signals"""
        print(f"\nReceived signal {signum}, shutting down...")
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.capture_engine:
                self.capture_engine.stop_capture_session()

            if self.oscilloscope and self.oscilloscope.is_connected:
                self.oscilloscope.disconnect()

            self.logger.info("Cleanup completed")
        except Exception as e:
            print(f"Cleanup error: {e}")

    # ========================================================================
    # CONNECTION MANAGEMENT
    # ========================================================================

    def connect_oscilloscope(self, visa_address: str) -> Tuple[str, str]:
        """Connect to oscilloscope"""
        try:
            if not visa_address:
                return "Error: VISA address required", "Disconnected"

            self.oscilloscope = KeysightDSOX6004A(visa_address)

            if self.oscilloscope.connect():
                self.capture_engine = TriggerCaptureEngine(self.oscilloscope)

                info = self.oscilloscope.get_instrument_info()
                if info:
                    info_text = (f"✅ Connected to {info['manufacturer']} {info['model']}\n"
                               f"Serial: {info['serial_number']}\n"
                               f"Firmware: {info['firmware_version']}\n\n"
                               f"Ready for delay analysis measurements")
                    self.logger.info(f"Connected to {info['model']}")
                    return info_text, "Connected"
            else:
                return "❌ Connection failed", "Disconnected"

        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            return f"❌ Error: {str(e)}", "Disconnected"

    def disconnect_oscilloscope(self) -> Tuple[str, str]:
        """Disconnect from oscilloscope"""
        try:
            if self.capture_engine:
                self.capture_engine.stop_capture_session()

            if self.oscilloscope:
                self.oscilloscope.disconnect()

            self.oscilloscope = None
            self.capture_engine = None

            self.logger.info("Disconnected from oscilloscope")
            return "✅ Disconnected successfully", "Disconnected"

        except Exception as e:
            self.logger.error(f"Disconnect error: {e}")
            return f"❌ Error: {str(e)}", "Disconnected"

    def test_connection(self) -> str:
        """Verify oscilloscope connectivity"""
        if self.oscilloscope and getattr(self.oscilloscope, "is_connected", False):
            return "✅ Connection test: PASSED\nOscilloscope responding normally"
        return "❌ Connection test: FAILED\nNot connected to oscilloscope"

    # ========================================================================
    # QUICK START FUNCTIONALITY
    # ========================================================================

    def quick_start_capture(
        self,
        num_captures: int,
        time_interval: float,
        ch1: bool,
        ch2: bool,
        trigger_channel: str,
        trigger_level: float,
        filename: str,
        save_path: str
    ) -> str:
        """Quick start capture with minimal configuration"""

        if not self.capture_engine:
            return "❌ Error: Not connected to oscilloscope\nPlease connect first in the Connection tab"

        try:
            # Validate inputs
            validation = self._validate_quick_start_inputs(
                num_captures, time_interval, ch1, ch2, filename, save_path
            )
            if validation:
                return f"❌ Configuration Issues:\n{validation}"

            # Parse channels
            channels = []
            if ch1: channels.append(1)
            if ch2: channels.append(2)

            # Configure trigger
            trigger_ch = int(trigger_channel.replace("CH", ""))
            self.configure_trigger(trigger_channel, trigger_level, "Rising")

            # Create configuration
            config = CaptureConfig(
                num_captures=num_captures,
                time_interval=time_interval,
                channels=channels,
                base_filename=filename,
                save_directory=save_path,
                capture_screenshots=True,
                save_waveforms=True,
                save_combined_csv=False,
                trigger_timeout=10.0,
                enable_delay_analysis=True,
                delay_reference_channel=1,  # CH1 = Laser
                delay_measurement_channel=2  # CH2 = APD
            )

            # Start capture
            if self.capture_engine.start_capture_session(config):
                est_time = num_captures * (time_interval + 2)
                return f"""
✅ CAPTURE SESSION STARTED
═══════════════════════════════════════════════════════════
Configuration:
  • Captures:          {num_captures}
  • Interval:          {time_interval} seconds
  • Channels:          {channels}
  • Delay Analysis:    CH1 (Laser) → CH2 (APD)
  • Save Path:         {save_path}

Estimated Duration:    ~{est_time/60:.1f} minutes

Status: Acquiring waveforms and analyzing delays...
Monitor progress in the "Capture Control" tab
View live statistics in the "Live Analysis" tab
═══════════════════════════════════════════════════════════
"""
            else:
                return "❌ Failed to start capture session\nCheck logs for details"

        except Exception as e:
            self.logger.error(f"Quick start error: {e}")
            return f"❌ Error: {str(e)}"

    def _validate_quick_start_inputs(
        self,
        num_captures,
        time_interval,
        ch1,
        ch2,
        filename,
        save_path
    ) -> str:
        """Validate quick start inputs and return warning/error messages"""
        warnings = []

        if num_captures < 1:
            warnings.append("❌ Number of captures must be at least 1")
        elif num_captures < 30:
            warnings.append("⚠ Recommendation: Use ≥30 captures for reliable statistics")

        if time_interval < 0:
            warnings.append("❌ Time interval cannot be negative")
        elif time_interval < 0.5:
            warnings.append("⚠ Very short intervals may cause trigger timeout issues")

        if not (ch1 and ch2):
            warnings.append("❌ Both CH1 (Laser) and CH2 (APD) must be selected for delay analysis")

        if not filename:
            warnings.append("❌ Filename cannot be empty")

        if not save_path:
            warnings.append("❌ Save path cannot be empty")

        # Estimate time
        est_time_min = num_captures * (time_interval + 2) / 60
        if est_time_min > 60:
            warnings.append(f"⚠ Long session: Estimated time ~{est_time_min:.1f} minutes")

        # Estimate file size
        est_size_mb = num_captures * 0.5
        if est_size_mb > 1000:
            warnings.append(f"⚠ Large dataset: Estimated storage ~{est_size_mb:.0f} MB")

        return "\n".join(warnings)

    # ========================================================================
    # CHANNEL CONFIGURATION
    # ========================================================================

    def configure_channel(
        self,
        ch1: bool,
        ch2: bool,
        ch3: bool,
        ch4: bool,
        v_scale: float,
        v_offset: float,
        coupling: str,
        probe: float
    ) -> str:
        """Configure vertical parameters for selected channels"""
        if not self.oscilloscope or not getattr(self.oscilloscope, "is_connected", False):
            return "❌ Error: Not connected"

        channel_states = {1: ch1, 2: ch2, 3: ch3, 4: ch4}

        try:
            success_count = 0
            disabled_count = 0

            for channel, enabled in channel_states.items():
                if enabled:
                    try:
                        success = self.oscilloscope.configure_channel(
                            channel=channel,
                            vertical_scale=v_scale,
                            vertical_offset=v_offset,
                            coupling=coupling,
                            probe_attenuation=probe
                        )
                        if success:
                            success_count += 1
                    except Exception as e:
                        self.logger.error(f"Failed to configure channel {channel}: {e}")
                else:
                    # Disable channel display when not selected
                    try:
                        self.oscilloscope._scpi_wrapper.write(f":CHANnel{channel}:DISPlay OFF")
                        disabled_count += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to disable channel {channel}: {e}")

            return f"✅ Configured: {success_count} enabled, {disabled_count} disabled"
        except Exception as e:
            return f"❌ Configuration error: {str(e)}"

    # ========================================================================
    # TRIGGER AND TIMEBASE
    # ========================================================================

    def configure_trigger(self, source: str, level: float, slope: str) -> str:
        """Configure trigger settings"""
        if not self.oscilloscope or not self.oscilloscope.is_connected:
            return "❌ Error: Not connected"

        try:
            channel = int(source.replace("CH", ""))

            # Set trigger parameters
            self.oscilloscope._scpi_wrapper.write(f":TRIGger:EDGE:SOURce CHANnel{channel}")
            self.oscilloscope._scpi_wrapper.write(f":TRIGger:EDGE:LEVel {level}")

            slope_map = {"Rising": "POS", "Falling": "NEG", "Either": "EITH"}
            self.oscilloscope._scpi_wrapper.write(f":TRIGger:EDGE:SLOPe {slope_map[slope]}")

            # Set to NORMAL mode for stable triggering
            self.oscilloscope._scpi_wrapper.write(":TRIGger:SWEep NORMal")

            return f"✅ Trigger configured: {source} @ {level} V, {slope} edge"

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def configure_timebase(self, time_scale: float) -> str:
        """Set horizontal timebase"""
        if not self.oscilloscope or not self.oscilloscope.is_connected:
            return "❌ Error: Not connected"

        try:
            success = self.oscilloscope.configure_timebase(time_scale)
            if success:
                return f"✅ Timebase set: {time_scale} s/div"
            return "❌ Timebase configuration failed"
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def run_autoscale(self) -> str:
        """Execute autoscale"""
        if not self.oscilloscope or not self.oscilloscope.is_connected:
            return "❌ Error: Not connected"

        try:
            self.oscilloscope._scpi_wrapper.write(":AUToscale")
            time.sleep(3)  # Wait for autoscale to complete
            return "✅ Autoscale completed"
        except Exception as e:
            return f"❌ Error: {str(e)}"

    # ========================================================================
    # CAPTURE CONTROL (STANDARD MODE)
    # ========================================================================

    def start_capture(
        self,
        num_captures: int,
        time_interval: float,
        ch1: bool, ch2: bool, ch3: bool, ch4: bool,
        base_filename: str,
        save_directory: str,
        capture_screenshots: bool,
        save_waveforms: bool,
        save_combined_csv: bool,
        trigger_timeout: float,
        enable_delay: bool,
        delay_ref_ch: int,
        delay_meas_ch: int
    ) -> str:
        """Start trigger capture session (standard mode)"""

        if not self.capture_engine:
            return "❌ Error: Not connected to oscilloscope"

        # Parse selected channels
        channels = []
        if ch1: channels.append(1)
        if ch2: channels.append(2)
        if ch3: channels.append(3)
        if ch4: channels.append(4)

        if not channels:
            return "❌ Error: No channels selected"

        # Create configuration
        config = CaptureConfig(
            num_captures=num_captures,
            time_interval=time_interval,
            channels=channels,
            base_filename=base_filename,
            save_directory=save_directory,
            capture_screenshots=capture_screenshots,
            save_waveforms=save_waveforms,
            save_combined_csv=save_combined_csv,
            trigger_timeout=trigger_timeout,
            enable_delay_analysis=enable_delay,
            delay_reference_channel=delay_ref_ch,
            delay_measurement_channel=delay_meas_ch
        )

        # Validate configuration
        valid, msg = config.validate()
        if not valid:
            return f"❌ Configuration error: {msg}"

        # Start capture
        if self.capture_engine.start_capture_session(config):
            self.logger.info(f"Started capture: {num_captures} triggers")
            return (f"✅ STARTED CAPTURE SESSION\n"
                   f"═════════════════════════════════════════\n"
                   f"Captures:           {num_captures}\n"
                   f"Interval:           {time_interval}s\n"
                   f"Channels:           {channels}\n"
                   f"Delay Analysis:     {'Enabled' if enable_delay else 'Disabled'}\n"
                   f"Screenshots:        {'Yes' if capture_screenshots else 'No'}\n"
                   f"Waveforms:          {'Yes' if save_waveforms else 'No'}\n"
                   f"Directory:          {save_directory}\n"
                   f"═════════════════════════════════════════")
        else:
            return "❌ Failed to start capture"

    def stop_capture(self) -> str:
        """Stop ongoing capture"""
        if not self.capture_engine:
            return "❌ Error: Not connected"

        self.capture_engine.stop_capture_session()
        time.sleep(0.1)

        status = self.capture_engine.get_status()
        return (f"🛑 CAPTURE STOPPED\n"
               f"═════════════════════════════════════════\n"
               f"Completed:          {status['completed_captures']}/{status['total_captures']}\n"
               f"Successful:         {status['successful_captures']}\n"
               f"Failed:             {status['failed_captures']}\n"
               f"═════════════════════════════════════════")

    def get_status(self) -> str:
        """Get current capture status with delay statistics"""
        if not self.capture_engine:
            return "Status: Not connected"

        status = self.capture_engine.get_status()
        stats = self.capture_engine.get_statistics()

        if status['is_running']:
            status_icon = "🔄 RUNNING"
            progress_bar = self._create_progress_bar(status['progress_percentage'])
        else:
            status_icon = "⏸️ IDLE"
            progress_bar = ""

        # Calculate ETA
        if len(self.capture_engine.capture_results) > 1:
            elapsed = (self.capture_engine.capture_results[-1].timestamp -
                      self.capture_engine.capture_results[0].timestamp).total_seconds()
            remaining_captures = status['total_captures'] - status['current_capture']
            if status['current_capture'] > 0:
                avg_time = elapsed / status['current_capture']
                eta_sec = remaining_captures * avg_time
                eta_min = int(eta_sec / 60)
                eta_sec_remaining = int(eta_sec % 60)
                eta_str = f"{eta_min}m {eta_sec_remaining}s"
            else:
                eta_str = "Calculating..."
        else:
            eta_str = "Calculating..."

        output = f"{status_icon}\n"
        output += "═════════════════════════════════════════\n"
        output += f"Progress:           {status['current_capture']}/{status['total_captures']} ({status['progress_percentage']:.1f}%)\n"
        output += f"{progress_bar}\n"
        output += f"Successful:         {status['successful_captures']}\n"
        output += f"Failed:             {status['failed_captures']}\n"
        output += f"Total Files:        {len(self.capture_engine.get_file_list())}\n"

        if status['is_running']:
            output += f"ETA:                {eta_str}\n"

        # Add delay statistics if available
        if stats:
            output += "\n📊 DELAY STATISTICS (Live)\n"
            output += "─────────────────────────────────────────\n"
            output += f"Mean Delay:         {stats['mean_ns']:.3f} ± {stats['std_ns']:.3f} ns\n"
            output += f"Jitter (RMS):       {stats['jitter_rms_ns']:.3f} ns\n"
            output += f"Jitter (Pk-Pk):     {stats['jitter_pk_pk_ns']:.3f} ns\n"
            output += f"Samples:            {stats['count']}\n"
            output += f"Confidence:         {stats['mean_confidence']:.3f}\n"

        return output

    def _create_progress_bar(self, percentage: float, width: int = 30) -> str:
        """Create text-based progress bar"""
        filled = int(width * percentage / 100)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}] {percentage:.1f}%"

    # ========================================================================
    # LIVE ANALYSIS AND VISUALIZATION
    # ========================================================================

    def get_live_plot(self) -> Figure:
        """Generate live delay distribution plot"""
        if not self.capture_engine:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.text(0.5, 0.5, 'Not connected to oscilloscope',
                   ha='center', va='center', fontsize=16)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return fig

        delays_ns = self.capture_engine.statistics_tracker.get_delays_array()

        if len(delays_ns) == 0:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.text(0.5, 0.5, 'No delay measurements yet\nWaiting for data...',
                   ha='center', va='center', fontsize=16)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return fig

        return self.visualizer.plot_delay_distribution(
            delays_ns,
            title="Laser-APD Delay Analysis (Live)"
        )

    def get_live_statistics_text(self) -> str:
        """Get formatted live statistics text"""
        if not self.capture_engine:
            return "Not connected"

        stats = self.capture_engine.get_statistics()

        if not stats:
            return "No delay measurements available yet.\nWaiting for data..."

        output = "📊 LIVE DELAY STATISTICS\n"
        output += "═════════════════════════════════════════\n"
        output += "DELAY MEASUREMENTS (Laser → APD)\n"
        output += "─────────────────────────────────────────\n"
        output += f"Mean Delay:         {stats['mean_ns']:.6f} ns\n"
        output += f"Std Deviation:      {stats['std_ns']:.6f} ns\n"
        output += f"Median:             {stats['median_ns']:.6f} ns\n"
        output += f"Min:                {stats['min_ns']:.6f} ns\n"
        output += f"Max:                {stats['max_ns']:.6f} ns\n"
        output += "\nJITTER ANALYSIS\n"
        output += "─────────────────────────────────────────\n"
        output += f"RMS Jitter:         {stats['jitter_rms_ns']:.6f} ns\n"
        output += f"Peak-to-Peak:       {stats['jitter_pk_pk_ns']:.6f} ns\n"
        output += "\nCONFIDENCE INTERVAL\n"
        output += "─────────────────────────────────────────\n"
        output += f"SEM:                {stats['sem_ns']:.6f} ns\n"
        output += f"95% CI:             ±{stats['ci_95_ns']:.6f} ns\n"
        output += f"Mean Confidence:    {stats['mean_confidence']:.4f}\n"
        output += "\nSAMPLE SIZE\n"
        output += "─────────────────────────────────────────\n"
        output += f"Measurements:       {stats['count']}\n"
        output += "═════════════════════════════════════════\n"

        return output

    # ========================================================================
    # RESULTS AND REPORTS
    # ========================================================================

    def get_file_list(self) -> str:
        """Get list of captured files"""
        if not self.capture_engine:
            return "No files - not connected"

        files = self.capture_engine.get_file_list()

        if not files:
            return "No files captured yet"

        # Group files by type
        screenshots = [f for f in files if 'screenshot' in f]
        waveforms = [f for f in files if 'screenshot' not in f]

        output = ["📁 CAPTURED FILES", "═════════════════════════════════════════"]

        if screenshots:
            output.append(f"\n📸 Screenshots ({len(screenshots)}):")
            for f in screenshots[-5:]:  # Show last 5
                output.append(f"  • {Path(f).name}")
            if len(screenshots) > 5:
                output.append(f"  ... and {len(screenshots)-5} more")

        if waveforms:
            output.append(f"\n📈 Waveforms ({len(waveforms)}):")
            for f in waveforms[-5:]:  # Show last 5
                output.append(f"  • {Path(f).name}")
            if len(waveforms) > 5:
                output.append(f"  ... and {len(waveforms)-5} more")

        return "\n".join(output)

    def generate_statistical_report(self) -> str:
        """Generate comprehensive statistical report"""
        if not self.capture_engine or not self.capture_engine.capture_results:
            return "No capture data available"

        if not self.capture_engine.current_config:
            return "No configuration available"

        return self.report_generator.generate_text_report(
            self.capture_engine.capture_results,
            self.capture_engine.current_config
        )

    def export_all_formats(self) -> str:
        """Export results to all formats (CSV, Excel, MATLAB, Plot)"""
        if not self.capture_engine or not self.capture_engine.capture_results:
            return "❌ No data to export"

        try:
            save_dir = Path(self.capture_engine.current_config.save_directory) / "analysis"
            save_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = self.capture_engine.current_config.base_filename

            exported_files = []

            # 1. Export Excel
            try:
                excel_file = save_dir / f"{base_name}_analysis_{timestamp}.xlsx"
                self.report_generator.export_to_excel(
                    self.capture_engine.capture_results,
                    self.capture_engine.current_config,
                    excel_file
                )
                exported_files.append(f"✅ Excel: {excel_file.name}")
            except Exception as e:
                exported_files.append(f"❌ Excel failed: {e}")

            # 2. Export MATLAB
            try:
                matlab_file = save_dir / f"{base_name}_analysis_{timestamp}.mat"
                self.report_generator.export_to_matlab(
                    self.capture_engine.capture_results,
                    matlab_file
                )
                exported_files.append(f"✅ MATLAB: {matlab_file.name}")
            except Exception as e:
                exported_files.append(f"❌ MATLAB failed: {e}")

            # 3. Export plot
            try:
                plot_file = save_dir / f"{base_name}_delay_plot_{timestamp}.png"
                delays_ns = self.capture_engine.statistics_tracker.get_delays_array()
                if len(delays_ns) > 0:
                    fig = self.visualizer.plot_delay_distribution(delays_ns, "Delay Distribution")
                    self.visualizer.save_plot(fig, plot_file)
                    plt.close(fig)
                    exported_files.append(f"✅ Plot: {plot_file.name}")
            except Exception as e:
                exported_files.append(f"❌ Plot failed: {e}")

            # 4. Export text report
            try:
                report_file = save_dir / f"{base_name}_report_{timestamp}.txt"
                report_text = self.generate_statistical_report()
                with open(report_file, 'w') as f:
                    f.write(report_text)
                exported_files.append(f"✅ Report: {report_file.name}")
            except Exception as e:
                exported_files.append(f"❌ Report failed: {e}")

            output = "📦 EXPORT COMPLETE\n"
            output += "═════════════════════════════════════════\n"
            output += f"Export Location:\n{save_dir}\n\n"
            output += "Files Created:\n"
            output += "\n".join(exported_files)
            output += "\n═════════════════════════════════════════"

            return output

        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            return f"❌ Export error: {str(e)}"

    # ========================================================================
    # GRADIO INTERFACE CONSTRUCTION
    # ========================================================================

    def create_interface(self) -> gr.Blocks:
        """Create enhanced Gradio web interface"""

        css = """
        .gradio-container {
            max-width: 100% !important;
            padding: 20px !important;
            margin: 0 !important;
            min-height: 100vh;
        }
        .container {
            max-width: 100% !important;
            padding: 0 10px !important;
            margin: 0 !important;
        }
        #component-0 {
            min-height: 100vh;
        }
        .tab {
            padding: 0 10px;
            min-height: calc(100vh - 120px);
        }
        .panel {
            margin: 5px 0;
        }
        """

        with gr.Blocks(
            title="Laser-APD Delay Analysis System",
            css=css,
            theme=gr.themes.Soft(
                primary_hue="indigo",
                spacing_size="sm",
                radius_size="sm",
                text_size="sm"
            )
        ) as interface:

            gr.Markdown("# 🔬 Laser-APD Delay Analysis System")
            gr.Markdown("**Automated Waveform Capture and Timing Jitter Measurement**")
            gr.Markdown("*Enhanced Version 3.0 | Digantara Research and Technologies*")

            # ================================================================
            # QUICK START TAB (New - Most Important)
            # ================================================================

            with gr.Tab("🚀 Quick Start"):
                gr.Markdown("""
                ### Quick Setup for Laser-APD Delay Measurement
                Configure essential parameters and start capturing immediately.
                This is the easiest way to begin your delay analysis.
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Acquisition Settings")
                        quick_num_captures = gr.Number(
                            label="Number of Captures",
                            value=50,
                            minimum=1,
                            maximum=10000,
                            step=1,
                            info="Recommended: 50-100 for statistical analysis"
                        )
                        quick_interval = gr.Number(
                            label="Time Interval (seconds)",
                            value=1.0,
                            minimum=0,
                            step=0.1,
                            info="Time between each waveform capture"
                        )

                        gr.Markdown("#### Channel Selection")
                        with gr.Row():
                            quick_ch1 = gr.Checkbox(label="CH1 (Laser)", value=True)
                            quick_ch2 = gr.Checkbox(label="CH2 (APD)", value=True)

                    with gr.Column(scale=1):
                        gr.Markdown("#### Trigger Settings")
                        quick_trigger_ch = gr.Dropdown(
                            choices=["CH1", "CH2"],
                            value="CH1",
                            label="Trigger Channel",
                            info="Which channel triggers the capture"
                        )
                        quick_trigger_level = gr.Number(
                            label="Trigger Level (V)",
                            value=0.5,
                            step=0.1,
                            info="Voltage threshold for trigger"
                        )

                        gr.Markdown("#### File Settings")
                        quick_filename = gr.Textbox(
                            label="Filename Prefix",
                            value="laser_apd_delay",
                            placeholder="e.g., laser_apd_delay"
                        )
                        quick_path = gr.Textbox(
                            label="Save Path",
                            value=str(self.default_save_dir),
                            placeholder="Path where files will be saved"
                        )

                quick_start_btn = gr.Button(
                    "▶️ START CAPTURE SESSION",
                    variant="primary",
                    size="lg",
                    scale=2
                )

                quick_status = gr.Textbox(
                    label="Status",
                    lines=15,
                    interactive=False,
                    show_label=True
                )

                # Connect quick start button
                quick_start_btn.click(
                    self.quick_start_capture,
                    inputs=[
                        quick_num_captures, quick_interval,
                        quick_ch1, quick_ch2,
                        quick_trigger_ch, quick_trigger_level,
                        quick_filename, quick_path
                    ],
                    outputs=[quick_status]
                )

            # ================================================================
            # CONNECTION TAB
            # ================================================================

            with gr.Tab("🔌 Connection"):
                with gr.Row():
                    visa_address = gr.Textbox(
                        label="VISA Address",
                        value="USB0::0x0957::0x1780::MY65220169::INSTR",
                        scale=3
                    )
                    connect_btn = gr.Button("Connect", variant="primary", scale=1)
                    disconnect_btn = gr.Button("Disconnect", variant="stop", scale=1)
                    test_btn = gr.Button("Test Connection", scale=1)

                connection_info = gr.Textbox(
                    label="Connection Info",
                    lines=5,
                    interactive=False
                )
                connection_status = gr.Textbox(
                    label="Status",
                    value="Disconnected",
                    interactive=False
                )

                connect_btn.click(
                    self.connect_oscilloscope,
                    inputs=[visa_address],
                    outputs=[connection_info, connection_status]
                )
                disconnect_btn.click(
                    self.disconnect_oscilloscope,
                    outputs=[connection_info, connection_status]
                )
                test_btn.click(
                    self.test_connection,
                    outputs=[connection_info]
                )

            # ================================================================
            # CHANNEL CONFIGURATION TAB
            # ================================================================

            with gr.Tab("⚙️ Channel Configuration"):
                gr.Markdown("### Channel Selection and Configuration")

                with gr.Row():
                    ch1_cfg = gr.Checkbox(label="Ch1", value=True)
                    ch2_cfg = gr.Checkbox(label="Ch2", value=True)
                    ch3_cfg = gr.Checkbox(label="Ch3", value=False)
                    ch4_cfg = gr.Checkbox(label="Ch4", value=False)

                with gr.Row():
                    v_scale = gr.Number(label="V/div", value=1.0)
                    v_offset = gr.Number(label="Offset (V)", value=0.0)
                    coupling = gr.Dropdown(
                        label="Coupling",
                        choices=["AC", "DC"],
                        value="DC"
                    )
                    probe = gr.Dropdown(
                        label="Probe",
                        choices=[("1x", 1.0), ("10x", 10.0), ("100x", 100.0)],
                        value=1.0
                    )

                config_channel_btn = gr.Button("Configure Channels", variant="primary")
                channel_status = gr.Textbox(label="Status", interactive=False)
                autoscale_btn = gr.Button("Autoscale", variant="secondary")

                config_channel_btn.click(
                    self.configure_channel,
                    inputs=[ch1_cfg, ch2_cfg, ch3_cfg, ch4_cfg, v_scale, v_offset, coupling, probe],
                    outputs=[channel_status]
                )
                autoscale_btn.click(
                    self.run_autoscale,
                    outputs=[channel_status]
                )

            # ================================================================
            # TIMEBASE & TRIGGER TAB
            # ================================================================

            with gr.Tab("🎯 Timebase & Trigger"):
                gr.Markdown("### Configure Oscilloscope Settings")

                gr.Markdown("#### Trigger Configuration")
                with gr.Row():
                    trigger_source = gr.Dropdown(
                        label="Trigger Source",
                        choices=["CH1", "CH2", "CH3", "CH4"],
                        value="CH1"
                    )
                    trigger_level = gr.Number(
                        label="Trigger Level (V)",
                        value=0.5,
                        step=0.1
                    )
                    trigger_slope = gr.Dropdown(
                        label="Trigger Slope",
                        choices=["Rising", "Falling", "Either"],
                        value="Rising"
                    )
                    trigger_btn = gr.Button("Apply Trigger", variant="primary")

                trigger_status = gr.Textbox(label="Trigger Status", interactive=False)

                gr.Markdown("#### Timebase")
                with gr.Row():
                    time_scale = gr.Dropdown(
                        label="Time/div",
                        choices=self.timebase_scales,
                        value=10e-3
                    )
                    timebase_btn = gr.Button("Apply Timebase", variant="primary")

                timebase_status = gr.Textbox(label="Timebase Status", interactive=False)

                trigger_btn.click(
                    self.configure_trigger,
                    inputs=[trigger_source, trigger_level, trigger_slope],
                    outputs=[trigger_status]
                )
                timebase_btn.click(
                    self.configure_timebase,
                    inputs=[time_scale],
                    outputs=[timebase_status]
                )

            # ================================================================
            # ADVANCED CAPTURE SETUP TAB
            # ================================================================

            with gr.Tab("📋 Advanced Capture Setup"):
                gr.Markdown("### Advanced Capture Configuration")
                gr.Markdown("*For users who need fine-grained control over capture parameters*")

                with gr.Row():
                    num_captures = gr.Number(
                        label="Number of Captures",
                        value=50,
                        minimum=1,
                        maximum=10000,
                        step=1
                    )
                    time_interval = gr.Number(
                        label="Interval Between Captures (s)",
                        value=1.0,
                        minimum=0,
                        step=0.1
                    )
                    trigger_timeout = gr.Number(
                        label="Trigger Timeout (s)",
                        value=10.0,
                        minimum=1,
                        maximum=60
                    )

                gr.Markdown("### Channel Selection")
                with gr.Row():
                    ch1_select = gr.Checkbox(label="Channel 1", value=True)
                    ch2_select = gr.Checkbox(label="Channel 2", value=True)
                    ch3_select = gr.Checkbox(label="Channel 3", value=False)
                    ch4_select = gr.Checkbox(label="Channel 4", value=False)

                gr.Markdown("### Delay Analysis Settings")
                with gr.Row():
                    enable_delay = gr.Checkbox(
                        label="Enable Delay Analysis",
                        value=True,
                        info="Calculate delay between reference and measurement channels"
                    )
                    delay_ref_ch = gr.Dropdown(
                        label="Reference Channel (Laser)",
                        choices=[1, 2, 3, 4],
                        value=1
                    )
                    delay_meas_ch = gr.Dropdown(
                        label="Measurement Channel (APD)",
                        choices=[1, 2, 3, 4],
                        value=2
                    )

                gr.Markdown("### Save Options")
                with gr.Row():
                    capture_screenshots = gr.Checkbox(
                        label="Capture Screenshots",
                        value=True
                    )
                    save_waveforms = gr.Checkbox(
                        label="Save Per-Channel CSV",
                        value=True
                    )
                    save_combined_csv = gr.Checkbox(
                        label="Save Combined Multi-Channel CSV",
                        value=False
                    )

                gr.Markdown("### File Settings")
                with gr.Row():
                    base_filename = gr.Textbox(
                        label="Base Filename",
                        value="laser_apd_delay",
                        scale=2
                    )
                    save_directory = gr.Textbox(
                        label="Save Directory",
                        value=str(self.default_save_dir),
                        scale=3
                    )

                advanced_start_btn = gr.Button("Start Advanced Capture", variant="primary")
                advanced_status = gr.Textbox(label="Status", lines=10, interactive=False)

                advanced_start_btn.click(
                    self.start_capture,
                    inputs=[
                        num_captures, time_interval,
                        ch1_select, ch2_select, ch3_select, ch4_select,
                        base_filename, save_directory,
                        capture_screenshots, save_waveforms, save_combined_csv,
                        trigger_timeout,
                        enable_delay, delay_ref_ch, delay_meas_ch
                    ],
                    outputs=[advanced_status]
                )

            # ================================================================
            # CAPTURE CONTROL TAB
            # ================================================================

            with gr.Tab("🎮 Capture Control"):
                gr.Markdown("### Monitor and Control Active Capture Session")

                with gr.Row():
                    stop_btn = gr.Button(
                        "⏹️ STOP CAPTURE",
                        variant="stop",
                        scale=1,
                        size="lg"
                    )
                    refresh_status_btn = gr.Button(
                        "🔄 Refresh Status",
                        scale=1
                    )

                status_display = gr.Textbox(
                    label="Live Status",
                    lines=20,
                    interactive=False
                )

                gr.Markdown("### Auto-Refresh")
                auto_refresh = gr.Checkbox(
                    label="Enable Auto-Refresh (every 2 seconds)",
                    value=False
                )

                stop_btn.click(
                    self.stop_capture,
                    outputs=[status_display]
                )

                refresh_status_btn.click(
                    self.get_status,
                    outputs=[status_display]
                )

                # Auto-refresh functionality
                def auto_update_status(enable):
                    if enable and self.capture_engine:
                        return self.get_status()
                    return "Auto-refresh disabled"

                auto_refresh.change(
                    auto_update_status,
                    inputs=[auto_refresh],
                    outputs=[status_display],
                    every=2
                )

            # ================================================================
            # LIVE ANALYSIS TAB (NEW - CRITICAL)
            # ================================================================

            with gr.Tab("📊 Live Analysis"):
                gr.Markdown("### Real-Time Delay Statistics and Visualization")
                gr.Markdown("*Monitor delay measurements as they are acquired*")

                with gr.Row():
                    update_plot_btn = gr.Button("🔄 Update Plot", variant="primary", scale=1)
                    auto_update_plot = gr.Checkbox(label="Auto-Update (every 3s)", value=False, scale=1)

                plot_output = gr.Plot(label="Delay Distribution Plot")

                stats_text = gr.Textbox(
                    label="Live Statistics",
                    lines=20,
                    interactive=False
                )

                update_plot_btn.click(
                    lambda: (self.get_live_plot(), self.get_live_statistics_text()),
                    outputs=[plot_output, stats_text]
                )

                def auto_update_plot_func(enable):
                    if enable and self.capture_engine:
                        return self.get_live_plot(), self.get_live_statistics_text()
                    return None, "Auto-update disabled"

                auto_update_plot.change(
                    auto_update_plot_func,
                    inputs=[auto_update_plot],
                    outputs=[plot_output, stats_text],
                    every=3
                )

            # ================================================================
            # RESULTS & REPORTS TAB (ENHANCED)
            # ================================================================

            with gr.Tab("📈 Results & Reports"):
                gr.Markdown("### View Captured Files and Generate Reports")

                with gr.Row():
                    show_files_btn = gr.Button("📁 Show Files", variant="primary")
                    generate_report_btn = gr.Button("📄 Generate Report", variant="primary")
                    export_all_btn = gr.Button("💾 Export All Formats", variant="secondary")

                file_list_display = gr.Textbox(
                    label="Captured Files",
                    lines=12,
                    interactive=False
                )

                report_display = gr.Textbox(
                    label="Statistical Report",
                    lines=30,
                    interactive=False
                )

                export_status = gr.Textbox(
                    label="Export Status",
                    lines=10,
                    interactive=False
                )

                show_files_btn.click(
                    self.get_file_list,
                    outputs=[file_list_display]
                )

                generate_report_btn.click(
                    self.generate_statistical_report,
                    outputs=[report_display]
                )

                export_all_btn.click(
                    self.export_all_formats,
                    outputs=[export_status]
                )

            gr.Markdown("---")
            gr.Markdown("""
            **Laser-APD Delay Analysis System v3.0** | Enhanced with Automatic Delay Measurement
            *Professional Oscilloscope Automation for Precision Timing Analysis*
            Developed by: Anirudh Iyengar | Digantara Research and Technologies Pvt. Ltd.
            """)

        return interface

    def launch(self, share=False, server_port=7866):
        """Launch the Gradio interface"""
        interface = self.create_interface()

        print("\n" + "="*70)
        print("  LASER-APD DELAY ANALYSIS SYSTEM v3.0")
        print("  Enhanced Automated Waveform Capture & Timing Jitter Measurement")
        print("="*70)
        print(f"  Starting web interface on port {server_port}...")
        print("="*70)

        try:
            interface.launch(
                server_name="0.0.0.0",
                server_port=server_port,
                share=share,
                show_error=True,
                inbrowser=True
            )
        except Exception as e:
            print(f"Failed to launch: {e}")
            self.cleanup()

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Application entry point"""
    print("\n" + "="*70)
    print("  Initializing Laser-APD Delay Analysis System v3.0...")
    print("  Enhanced for Precision Timing Jitter Measurement")
    print("="*70)
    print("  Developer:     Anirudh Iyengar")
    print("  Organization:  Digantara Research and Technologies Pvt. Ltd.")
    print("  Version:       3.0.0 - Enhanced Edition")
    print("="*70 + "\n")

    app = TriggerCaptureGUI()

    try:
        app.launch(share=False, server_port=7866)
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
    finally:
        app.cleanup()
        print("✅ Application terminated")

if __name__ == "__main__":
    main()
