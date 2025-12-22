
#!/usr/bin/env python3
"""
Continuous Trigger Capture Automation for Keysight DSOX6004A Oscilloscope

Focused automation for capturing screenshots and waveform data after each successful
trigger event. Uses SINGLE mode to capture one trigger at a time, save the display
screenshot and waveform data, then repeat for specified number of captures.

KEY FEATURES:
- SINGLE mode trigger capture
- Automatic screenshot after each trigger
- Waveform data export (CSV)
- Configurable number of captures and intervals
- Progress tracking with file listing
- Professional error handling

Author: Senior Instrumentation Engineer
Organization: Digantara Research and Technologies Pvt. Ltd.
Date: 2025-01-22
Version: 2.0.0
"""

import sys
import os
import time
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import signal
import atexit
import json

import numpy as np
import pandas as pd
import gradio as gr

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
        return True, "Configuration valid"

@dataclass
class CaptureResult:
    """Result from a single trigger capture"""
    index: int
    timestamp: datetime
    screenshot_file: Optional[str] = None
    waveform_files: List[str] = None
    measurements: Dict[str, Any] = None
    success: bool = True
    error_message: Optional[str] = None

# ============================================================================
# TRIGGER CAPTURE ENGINE
# ============================================================================

class TriggerCaptureEngine:
    """
    Core engine for continuous trigger-based capture using SINGLE mode.
    Captures screenshot and waveform data after each successful trigger.
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
        
        # Thread safety
        self.lock = threading.RLock()
        
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
        
        # Create save directory
        save_dir = Path(config.save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        # Start capture thread
        self.capture_thread = threading.Thread(
            target=self._capture_loop,
            args=(config,),
            daemon=True
        )
        self.capture_thread.start()
        
        self.logger.info(f"Started capture session: {config.num_captures} captures")
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
    
    def get_file_list(self) -> List[str]:
        """Get list of all captured files"""
        files = []
        for result in self.capture_results:
            if result.screenshot_file:
                files.append(result.screenshot_file)
            if result.waveform_files:
                files.extend(result.waveform_files)
        return files
    
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
                
                if not result.success:
                    self.logger.error(f"Capture {capture_idx + 1} failed: {result.error_message}")
                else:
                    self.logger.info(f"Capture {capture_idx + 1} completed successfully")
                
                # Wait so that each capture loop period approximates config.time_interval
                # (capture time + wait time ≈ Interval Between Captures)
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
            
            # Save waveform data if enabled
            if config.save_waveforms or config.save_combined_csv:
                waveform_files = self._save_waveform_data(config, capture_idx, timestamp)
                if waveform_files:
                    result.waveform_files = waveform_files
                    self.logger.debug(f"Waveforms saved: {len(waveform_files)} files")
            
            # Get measurements for selected channels
            result.measurements = self._get_measurements(config.channels)
            
            result.success = True
            return result
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            self.logger.error(f"Capture failed: {e}")
            return result
    
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
            filepath = Path(config.save_directory) / filename
            
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
    
    def _save_waveform_data(self, config: CaptureConfig, capture_idx: int, timestamp: datetime) -> List[str]:
        """Save waveform data from all configured channels"""
        saved_files: List[str] = []
        channel_traces: Dict[int, Tuple[List[float], List[float]]] = {}
        
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
                voltage_data = [(val - y_reference) * y_increment + y_origin for val in raw_data]
                time_data = [x_origin + i * x_increment for i in range(len(voltage_data))]
                
                # Store for potential combined CSV
                channel_traces[channel] = (time_data, voltage_data)
                
                # Save per-channel CSV if requested
                if config.save_waveforms:
                    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    filename = f"{config.base_filename}_CH{channel}_{capture_idx:04d}_{timestamp_str}.csv"
                    filepath = Path(config.save_directory) / filename
                    
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
        if config.save_combined_csv and channel_traces:
            try:
                # Use shortest trace length across channels to align data
                min_len = min(len(tr[0]) for tr in channel_traces.values())
                if min_len > 0:
                    # Reference time axis from first channel in list
                    first_channel = config.channels[0]
                    time_ref = channel_traces[first_channel][0][:min_len]
                    data = {
                        'Time (s)': time_ref
                    }
                    for ch, (t_data, v_data) in channel_traces.items():
                        data[f'CH{ch} (V)'] = v_data[:min_len]
                    
                    df_multi = pd.DataFrame(data)
                    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    filename = f"{config.base_filename}_MULTI_{capture_idx:04d}_{timestamp_str}.csv"
                    filepath = Path(config.save_directory) / filename
                    
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
        
        return saved_files
    
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
# GRADIO USER INTERFACE
# ============================================================================

class TriggerCaptureGUI:
    """
    Professional Gradio interface for continuous trigger capture.
    Focused on screenshot and data capture after each trigger event.
    """
    
    def __init__(self):
        self.oscilloscope = None
        self.capture_engine = None
        self.logger = self._setup_logging()
        
        # Default paths
        self.default_save_dir = Path.cwd() / "trigger_captures"
        self.default_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Common timebase scales (label, value) for UI (matching main Keysight GUI)
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
        logger = logging.getLogger('TriggerCapture')
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
                    info_text = (f"Connected to {info['manufacturer']} {info['model']}\n"
                               f"Serial: {info['serial_number']}\n"
                               f"Firmware: {info['firmware_version']}")
                    self.logger.info(f"Connected to {info['model']}")
                    return info_text, "Connected"
            else:
                return "Connection failed", "Disconnected"
                
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            return f"Error: {str(e)}", "Disconnected"
    
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
            return "Disconnected successfully", "Disconnected"
            
        except Exception as e:
            self.logger.error(f"Disconnect error: {e}")
            return f"Error: {str(e)}", "Disconnected"
    
    def test_connection(self) -> str:
        """Verify oscilloscope connectivity"""
        if self.oscilloscope and getattr(self.oscilloscope, "is_connected", False):
            return "Connection test: PASSED"
        return "Connection test: FAILED - Not connected"
    
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
            return "Error: Not connected"
        
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
            
            return f"Configured: {success_count} enabled, {disabled_count} disabled"
        except Exception as e:
            return f"Configuration error: {str(e)}"
    
    # ========================================================================
    # CAPTURE CONTROL
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
        trigger_timeout: float
    ) -> str:
        """Start trigger capture session"""
        
        if not self.capture_engine:
            return "Error: Not connected to oscilloscope"
        
        # Parse selected channels
        channels = []
        if ch1: channels.append(1)
        if ch2: channels.append(2)
        if ch3: channels.append(3)
        if ch4: channels.append(4)
        
        if not channels:
            return "Error: No channels selected"
        
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
            trigger_timeout=trigger_timeout
        )
        
        # Validate configuration
        valid, msg = config.validate()
        if not valid:
            return f"Configuration error: {msg}"
        
        # Start capture
        if self.capture_engine.start_capture_session(config):
            self.logger.info(f"Started capture: {num_captures} triggers")
            return (f"STARTED CAPTURE SESSION\n"
                   f"-----------------------------------------\n"
                   f"Captures: {num_captures}\n"
                   f"Interval: {time_interval}s\n"
                   f"Channels: {channels}\n"
                   f"Screenshots: {'Yes' if capture_screenshots else 'No'}\n"
                   f"Waveforms: {'Yes' if save_waveforms else 'No'}\n"
                   f"Directory: {save_directory}")
        else:
            return "Failed to start capture"
    
    def stop_capture(self) -> str:
        """Stop ongoing capture"""
        if not self.capture_engine:
            return "Error: Not connected"
        
        self.capture_engine.stop_capture_session()
        time.sleep(0.1)
        
        status = self.capture_engine.get_status()
        return (f"CAPTURE STOPPED\n"
               f"-----------------------------------------\n"
               f"Completed: {status['completed_captures']}/{status['total_captures']}\n"
               f"Successful: {status['successful_captures']}\n"
               f"Failed: {status['failed_captures']}")
    
    def get_status(self) -> str:
        """Get current capture status"""
        if not self.capture_engine:
            return "Status: Not connected"
        
        status = self.capture_engine.get_status()
        
        if status['is_running']:
            status_icon = "RUNNING"
            progress_bar = self._create_progress_bar(status['progress_percentage'])
        else:
            status_icon = "IDLE"
            progress_bar = ""
        
        return (f"{status_icon}\n"
               f"-----------------------------------------\n"
               f"Progress: {status['current_capture']}/{status['total_captures']} "
               f"({status['progress_percentage']:.1f}%)\n"
               f"{progress_bar}\n"
               f"Successful: {status['successful_captures']}\n"
               f"Failed: {status['failed_captures']}\n"
               f"Total Files: {len(self.capture_engine.get_file_list())}")
    
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
        
        output = ["CAPTURED FILES", "-----------------------------------------"]
        
        if screenshots:
            output.append(f"\nScreenshots ({len(screenshots)}):")
            for f in screenshots[-5:]:  # Show last 5
                output.append(f"  • {Path(f).name}")
            if len(screenshots) > 5:
                output.append(f"  ... and {len(screenshots)-5} more")
        
        if waveforms:
            output.append(f"\nWaveforms ({len(waveforms)}):")
            for f in waveforms[-5:]:  # Show last 5
                output.append(f"  • {Path(f).name}")
            if len(waveforms) > 5:
                output.append(f"  ... and {len(waveforms)-5} more")
        
        return "\n".join(output)
    
    def get_capture_summary(self) -> str:
        """Generate capture session summary"""
        if not self.capture_engine or not self.capture_engine.capture_results:
            return "No capture data available"
        
        results = self.capture_engine.capture_results
        
        # Calculate statistics
        total = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total - successful
        
        # Get timing info
        if results:
            duration = (results[-1].timestamp - results[0].timestamp).total_seconds()
            avg_time_per_capture = duration / max(total - 1, 1) if total > 1 else 0
        else:
            duration = 0
            avg_time_per_capture = 0
        
        # Count files
        total_screenshots = sum(1 for r in results if r.screenshot_file)
        total_waveforms = sum(len(r.waveform_files) for r in results if r.waveform_files)
        
        summary = [
            "CAPTURE SESSION SUMMARY",
            "-----------------------------------------",
            "",
            "STATISTICS:",
            f"  • Total Captures: {total}",
            f"  • Successful: {successful} ({(successful/total*100):.1f}%)",
            f"  • Failed: {failed}",
            f"  • Success Rate: {(successful/total*100):.1f}%",
            "",
            "TIMING:",
            f"  • Total Duration: {duration:.1f} seconds",
            f"  • Avg Time/Capture: {avg_time_per_capture:.2f} seconds",
            "",
            "FILES SAVED:",
            f"  • Screenshots: {total_screenshots}",
            f"  • Waveform Files: {total_waveforms}",
            f"  • Total Files: {total_screenshots + total_waveforms}",
        ]
        
        # Add recent measurements if available
        recent_results = [r for r in results[-5:] if r.success and r.measurements]
        if recent_results:
            summary.extend(["", "RECENT MEASUREMENTS:"])
            
            for result in recent_results:
                summary.append(f"\n  Capture {result.index + 1}:")
                for ch, meas in result.measurements.items():
                    if meas and 'FREQ' in meas:
                        freq = meas.get('FREQ', 0)
                        vpp = meas.get('VPP', 0)
                        summary.append(f"    {ch}: {freq:.2e} Hz, {vpp:.3f} Vpp")
        
        return "\n".join(summary)
    
    def save_summary_report(self, save_directory: str) -> str:
        """Save detailed summary report to JSON"""
        if not self.capture_engine or not self.capture_engine.capture_results:
            return "No data to save"
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_summary_{timestamp}.json"
            filepath = Path(save_directory) / filename
            
            # Prepare summary data
            results = self.capture_engine.capture_results
            summary_data = {
                'timestamp': datetime.now().isoformat(),
                'total_captures': len(results),
                'successful_captures': sum(1 for r in results if r.success),
                'failed_captures': sum(1 for r in results if not r.success),
                'captures': []
            }
            
            for result in results:
                capture_data = {
                    'index': result.index,
                    'timestamp': result.timestamp.isoformat(),
                    'success': result.success,
                    'screenshot': result.screenshot_file,
                    'waveforms': result.waveform_files or [],
                    'measurements': result.measurements or {},
                    'error': result.error_message
                }
                summary_data['captures'].append(capture_data)
            
            # Save to JSON
            with open(filepath, 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)
            
            return f"Summary saved: {filename}"
            
        except Exception as e:
            self.logger.error(f"Failed to save summary: {e}")
            return f"Error saving summary: {str(e)}"
    
    def _create_progress_bar(self, percentage: float, width: int = 20) -> str:
        """Create text-based progress bar"""
        filled = int(width * percentage / 100)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"
    
    # ========================================================================
    # SCOPE CONFIGURATION
    # ========================================================================
    
    def configure_trigger(self, source: str, level: float, slope: str) -> str:
        """Configure trigger settings"""
        if not self.oscilloscope or not self.oscilloscope.is_connected:
            return "Error: Not connected"
        
        try:
            channel = int(source.replace("CH", ""))
            
            # Set trigger parameters
            self.oscilloscope._scpi_wrapper.write(f":TRIGger:EDGE:SOURce CHANnel{channel}")
            self.oscilloscope._scpi_wrapper.write(f":TRIGger:EDGE:LEVel {level}")
            
            slope_map = {"Rising": "POS", "Falling": "NEG", "Either": "EITH"}
            self.oscilloscope._scpi_wrapper.write(f":TRIGger:EDGE:SLOPe {slope_map[slope]}")
            
            # Set to NORMAL mode for stable triggering
            self.oscilloscope._scpi_wrapper.write(":TRIGger:SWEep NORMal")
            
            return f"Trigger configured: {source} @ {level} V, {slope}"
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def configure_timebase(self, time_scale: float) -> str:
        """Set horizontal timebase"""
        if not self.oscilloscope or not self.oscilloscope.is_connected:
            return "Error: Not connected"
        
        try:
            # Use the oscilloscope helper so scaling/timeout handling matches main GUI
            success = self.oscilloscope.configure_timebase(time_scale)
            if success:
                return f"Timebase set: {time_scale} s/div"
            return "Timebase configuration failed"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def run_autoscale(self) -> str:
        """Execute autoscale"""
        if not self.oscilloscope or not self.oscilloscope.is_connected:
            return "Error: Not connected"
        
        try:
            self.oscilloscope._scpi_wrapper.write(":AUToscale")
            time.sleep(3)  # Wait for autoscale to complete
            return "Autoscale completed"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def perform_autoscale(self) -> str:
        """Execute automatic vertical and horizontal scaling (alias for run_autoscale)."""
        return self.run_autoscale()
    
    # ========================================================================
    # ACQUISITION CONTROL
    # ========================================================================
    
    def set_acquisition_mode(self, mode_type: str) -> str:
        """Set oscilloscope acquisition mode"""
        if not self.oscilloscope or not getattr(self.oscilloscope, "is_connected", False):
            return "Error: Not connected"
        
        try:
            success = self.oscilloscope.set_acquire_mode(mode_type)
            if success:
                return f"Acquisition mode: {mode_type}"
            return "Failed to set acquisition mode"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def set_acquisition_type(self, acq_type: str) -> str:
        """Set oscilloscope acquisition type"""
        if not self.oscilloscope or not getattr(self.oscilloscope, "is_connected", False):
            return "Error: Not connected"
        
        try:
            success = self.oscilloscope.set_acquire_type(acq_type)
            if success:
                return f"Acquisition type: {acq_type}"
            return "Failed to set acquisition type"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def set_acquisition_count(self, average_count: int) -> str:
        """Set number of acquisitions to average"""
        if not self.oscilloscope or not getattr(self.oscilloscope, "is_connected", False):
            return "Error: Not connected"
        
        try:
            if not (2 <= average_count <= 65536):
                return f"Error: Count must be 2-65536, got {average_count}"
            
            success = self.oscilloscope.set_acquire_count(int(average_count))
            if success:
                return f"Acquisition count: {int(average_count)} averages"
            return "Failed to set acquisition count"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def query_acquisition_info(self) -> str:
        """Query and display current acquisition parameters"""
        if not self.oscilloscope or not getattr(self.oscilloscope, "is_connected", False):
            return "Error: Not connected"
        
        try:
            info_lines = []
            
            mode = self.oscilloscope.get_acquire_mode()
            acq_type = self.oscilloscope.get_acquire_type()
            count = self.oscilloscope.get_acquire_count()
            sample_rate = self.oscilloscope.get_sample_rate()
            points = self.oscilloscope.get_acquire_points()
            
            if mode is not None:
                info_lines.append(f"Mode: {mode}")
            if acq_type is not None:
                info_lines.append(f"Type: {acq_type}")
            if count is not None:
                info_lines.append(f"Count: {count}")
            if sample_rate is not None:
                info_lines.append(f"Sample Rate: {sample_rate:.3e} Hz")
            if points is not None:
                info_lines.append(f"Acquired Points: {points}")
            
            return "\n".join(info_lines) if info_lines else "No acquisition info available"
        except Exception as e:
            return f"Error: {str(e)}"
    
    # ========================================================================
    # GRADIO INTERFACE
    # ========================================================================
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio web interface"""
        """Build comprehensive Gradio web interface with full-page layout"""

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
            title="Trigger Capture Automation",
            css=css,
            theme=gr.themes.Soft(
                primary_hue="purple",
                spacing_size="sm",
                radius_size="sm",
                text_size="sm"
            )
        ) as interface:
            
            gr.Markdown("# Continuous Trigger Capture System")
            gr.Markdown("Automated Screenshot & Data Capture on Each Trigger Event")
            gr.Markdown("Developed by: Anirudh Iyengar | Digantara Research and Technologies")
            
            # Connection Tab
            with gr.Tab("Connection"):
                with gr.Row():
                    visa_address = gr.Textbox(
                        label="VISA Address",
                        value="USB0::0x0957::0x1780::MY65220169::INSTR",
                        scale=3
                    )
                    connect_btn = gr.Button("Connect", variant="primary", scale=1)
                    disconnect_btn = gr.Button("Disconnect", variant="stop", scale=1)
                    test_btn = gr.Button("Test", scale=1)
                
                connection_info = gr.Textbox(
                    label="Connection Info",
                    lines=3,
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
            
            # Channel Configuration Tab
            with gr.Tab("Channel Configuration"):
                gr.Markdown("### Channel Selection and Configuration")
                
                with gr.Row():
                    ch1_cfg = gr.Checkbox(label="Ch1", value=True)
                    ch2_cfg = gr.Checkbox(label="Ch2", value=False)
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
                autoscale_channels_btn = gr.Button("Autoscale", variant="secondary")
                channel_system_status = gr.Textbox(label="System Status", interactive=False, lines=3)
                
                config_channel_btn.click(
                    self.configure_channel,
                    inputs=[ch1_cfg, ch2_cfg, ch3_cfg, ch4_cfg, v_scale, v_offset, coupling, probe],
                    outputs=[channel_status]
                )
                autoscale_channels_btn.click(
                    self.perform_autoscale,
                    outputs=[channel_system_status]
                )
            
            # Timebase and Trigger Tab
            with gr.Tab("Timebase & Trigger"):
                gr.Markdown("### Configure oscilloscope before starting capture")
                
                gr.Markdown("### Trigger Configuration")
                with gr.Row():
                    trigger_source = gr.Dropdown(
                        label="Trigger Source",
                        choices=["CH1", "CH2", "CH3", "CH4"],
                        value="CH1"
                    )
                    trigger_level = gr.Number(
                        label="Trigger Level (V)",
                        value=0.0,
                        step=0.1
                    )
                    trigger_slope = gr.Dropdown(
                        label="Trigger Slope",
                        choices=["Rising", "Falling", "Either"],
                        value="Rising"
                    )
                    trigger_btn = gr.Button("Apply Trigger", variant="primary")
                
                trigger_status = gr.Textbox(label="Trigger Status", interactive=False)
                
                gr.Markdown("### Timebase")
                with gr.Row():
                    time_scale = gr.Dropdown(
                        label="Time/div",
                        choices=self.timebase_scales,
                        value=10e-3
                    )
                    timebase_btn = gr.Button("Apply Timebase", variant="primary")
                    autoscale_btn = gr.Button("Autoscale", variant="secondary")
                
                timebase_status = gr.Textbox(label="Status", interactive=False)
                
                # Configure handlers
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
                autoscale_btn.click(
                    self.run_autoscale,
                    outputs=[timebase_status]
                )
            
            # # Acquisition Control Tab
            # with gr.Tab("Acquisition Control"):
            #     gr.Markdown("### Acquisition Mode")
            #     with gr.Row():
            #         acq_mode = gr.Dropdown(
            #             label="Mode",
            #             choices=["RTIMe", "ETIMe", "SEGMented"],
            #             value="RTIMe",
            #             info="RTIMe: Real-time, ETIMe: Equivalent-time, SEGMented: Multi-event capture"
            #         )
            #         acq_mode_btn = gr.Button("Apply Mode", variant="primary")
            #     acq_mode_status = gr.Textbox(label="Status", interactive=False)
            #     acq_mode_btn.click(
            #         self.set_acquisition_mode,
            #         inputs=[acq_mode],
            #         outputs=[acq_mode_status]
            #     )
                
            #     gr.Markdown("Acquisition Type")
            #     with gr.Row():
            #         acq_type = gr.Dropdown(
            #             label="Type",
            #             choices=["NORMal", "AVERage", "HRESolution", "PEAK"],
            #             value="NORMal",
            #             info=(
            #                 "NORMal: Standard, AVERage: Noise reduction, "
            #                 "HRESolution: Better resolution, PEAK: Transient capture"
            #             )
            #         )
            #         acq_type_btn = gr.Button("Apply Type", variant="primary")
            #     acq_type_status = gr.Textbox(label="Status", interactive=False)
            #     acq_type_btn.click(
            #         self.set_acquisition_type,
            #         inputs=[acq_type],
            #         outputs=[acq_type_status]
            #     )
                
            #     gr.Markdown("### Averaging Configuration")
            #     with gr.Row():
            #         avg_count = gr.Slider(
            #             label="Averaging Count",
            #             minimum=2,
            #             maximum=65536,
            #             value=16,
            #             step=1
            #         )
            #         avg_btn = gr.Button("Apply Averaging", variant="primary")
            #     avg_status = gr.Textbox(label="Status", interactive=False)
            #     avg_btn.click(
            #         self.set_acquisition_count,
            #         inputs=[avg_count],
            #         outputs=[avg_status]
            #     )
                
            #     gr.Markdown("### Acquisition Information")
            #     acq_info_btn = gr.Button("Query Info", variant="secondary")
            #     acq_info = gr.Textbox(label="Acquisition Info", interactive=False, lines=6)
            #     acq_info_btn.click(
            #         self.query_acquisition_info,
            #         outputs=[acq_info]
            #     )
            
            # Capture Configuration Tab
            with gr.Tab("Capture Setup"):
                gr.Markdown("### Capture Parameters")
                
                with gr.Row():
                    num_captures = gr.Number(
                        label="Number of Captures",
                        value=10,
                        minimum=1,
                        maximum=10000,
                        step=1,
                        info="How many trigger events to capture"
                    )
                    time_interval = gr.Number(
                        label="Interval Between Captures (s)",
                        value=1.0,
                        minimum=0,
                        step=0.1,
                        info="Target interval between captures (includes capture time where possible)"
                    )
                    trigger_timeout = gr.Number(
                        label="Trigger Timeout (s)",
                        value=10.0,
                        minimum=1,
                        maximum=60,
                        info="Max time to wait for trigger"
                    )
                
                gr.Markdown("### Channel Selection")
                with gr.Row():
                    ch1_select = gr.Checkbox(label="Channel 1", value=True)
                    ch2_select = gr.Checkbox(label="Channel 2", value=True)
                    ch3_select = gr.Checkbox(label="Channel 3", value=False)
                    ch4_select = gr.Checkbox(label="Channel 4", value=False)
                
                gr.Markdown("### Save Options")
                with gr.Row():
                    capture_screenshots = gr.Checkbox(
                        label="Capture Screenshots",
                        value=True,
                        info="Save oscilloscope display after each trigger"
                    )
                    save_waveforms = gr.Checkbox(
                        label="Save Waveform Data",
                        value=True,
                        info="Export waveform data to CSV files"
                    )
                    save_combined_csv = gr.Checkbox(
                        label="Save Combined Multi-Channel CSV",
                        value=False,
                        info="Save all selected channels into a single CSV per capture"
                    )
                
                gr.Markdown("### File Settings")
                with gr.Row():
                    base_filename = gr.Textbox(
                        label="Base Filename",
                        value="trigger_capture",
                        scale=2
                    )
                    save_directory = gr.Textbox(
                        label="Save Directory",
                        value=str(self.default_save_dir),
                        scale=3
                    )
                
                # Estimated time calculation
                def calculate_time(n, interval):
                    total = n * (interval + 2)  # Add ~2s for capture/save
                    return f"Estimated time: ~{total:.1f} seconds ({total/60:.1f} minutes)"
                
                time_estimate = gr.Textbox(
                    label="Time Estimate",
                    interactive=False
                )
                
                num_captures.change(
                    calculate_time,
                    inputs=[num_captures, time_interval],
                    outputs=[time_estimate]
                )
                time_interval.change(
                    calculate_time,
                    inputs=[num_captures, time_interval],
                    outputs=[time_estimate]
                )
            
            # Capture Control Tab
            with gr.Tab("Capture Control"):
                gr.Markdown("### Control Panel")
                
                with gr.Row():
                    start_btn = gr.Button(
                        "Start Capture",
                        variant="primary",
                        scale=2,
                        elem_classes=["large-button"]
                    )
                    stop_btn = gr.Button(
                        "Stop",
                        variant="stop",
                        scale=1
                    )
                
                capture_output = gr.Textbox(
                    label="Capture Output",
                    lines=8,
                    interactive=False
                )
                
                gr.Markdown("### Live Status")
                with gr.Row():
                    refresh_btn = gr.Button("Refresh", scale=1)
                    auto_refresh = gr.Checkbox(
                        label="Auto Refresh (2s)",
                        value=False
                    )
                
                status_display = gr.Textbox(
                    label="Current Status",
                    lines=7,
                    interactive=False
                )
                
                # Start capture
                start_btn.click(
                    self.start_capture,
                    inputs=[
                        num_captures, time_interval,
                        ch1_select, ch2_select, ch3_select, ch4_select,
                        base_filename, save_directory,
                        capture_screenshots, save_waveforms, save_combined_csv, trigger_timeout
                    ],
                    outputs=[capture_output]
                )
                
                # Stop capture
                stop_btn.click(
                    self.stop_capture,
                    outputs=[capture_output]
                )
                
                # Refresh status
                refresh_btn.click(
                    self.get_status,
                    outputs=[status_display]
                )
                
                # Auto-refresh
                def auto_update(enable):
                    if enable:
                        return self.get_status()
                    return "Auto-refresh disabled"
                
                auto_refresh.change(
                    auto_update,
                    inputs=[auto_refresh],
                    outputs=[status_display],
                    every=2  # Update every 2 seconds
                )
            
            # Results Tab
            with gr.Tab("Results"):
                gr.Markdown("### Captured Files")
                
                with gr.Row():
                    show_files_btn = gr.Button("Show Files", variant="primary")
                    summary_btn = gr.Button("Generate Summary", variant="primary")
                    save_report_btn = gr.Button("Save Report", variant="secondary")
                
                file_list_display = gr.Textbox(
                    label="File List",
                    lines=15,
                    interactive=False
                )
                
                summary_display = gr.Textbox(
                    label="Capture Summary",
                    lines=20,
                    interactive=False
                )
                
                report_status = gr.Textbox(
                    label="Report Status",
                    interactive=False
                )
                
                # File list
                show_files_btn.click(
                    self.get_file_list,
                    outputs=[file_list_display]
                )
                
                # Summary
                summary_btn.click(
                    self.get_capture_summary,
                    outputs=[summary_display]
                )
                
                # Save report
                save_report_btn.click(
                    lambda: self.save_summary_report(str(self.default_save_dir)),
                    outputs=[report_status]
                )
            
            gr.Markdown("---")
            gr.Markdown("**Trigger Capture System** v2.0 | Professional Oscilloscope Automation")
        
        return interface
    
    def launch(self, share=False, server_port=7866):
        """Launch the Gradio interface"""
        interface = self.create_interface()
        
        print("\n" + "="*60)
        print("TRIGGER CAPTURE AUTOMATION SYSTEM")
        print("Screenshot & Data Capture on Trigger Events")
        print("="*60)
        print(f"Starting web interface on port {server_port}...")
        
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
    print("Initializing Trigger Capture System...")
    print("Developed by: Anirudh Iyengar")
    print("Organization: Digantara Research and Technologies")
    print("-" * 60)
    
    app = TriggerCaptureGUI()
    
    try:
        app.launch(share=False, server_port=7866)
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        app.cleanup()
        print("Application terminated")

if __name__ == "__main__":
    main()
