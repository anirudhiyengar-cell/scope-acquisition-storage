# Continuous Trigger Capture System

## Automated Screenshot & Waveform Data Acquisition for Keysight Oscilloscope

**Developer**: Anirudh Iyengar
**Organization**: Digantara Research and Technologies Pvt. Ltd.
**Version**: 2.0.0
**Date**: January 22, 2025

---

## Overview

This system automates the process of capturing oscilloscope screenshots and waveform data on trigger events. It continuously monitors for trigger conditions, captures the display and raw data, and saves everything with proper timestamps.

**Key Features:**
- Trigger-based automated capture
- PNG screenshot generation
- CSV waveform data export
- Multi-channel support
- Real-time progress monitoring
- Web-based control interface
- Configurable capture parameters

---

## Quick Start

### Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Application

Launch the web interface:

```bash
python continuous_trigger_capture.py
```

The application will start a local web server and open in your default browser.

### Basic Workflow

1. Connect to the oscilloscope (Connection tab)
2. Configure channels (Channel Configuration tab)
3. Set trigger parameters (Timebase & Trigger tab)
4. Configure capture settings (Capture Setup tab)
5. Start capture (Capture Control tab)
6. View results (Results tab)

---

## Configuration

### 1. Oscilloscope Connection

Obtain the VISA address from your oscilloscope:
- Navigate to **Utility** → **I/O** → **LAN Configuration**
- Note the VISA address (format: `USB0::0x0957::0x1780::MY65220169::INSTR`)

In the web interface:
- Go to the Connection tab
- Enter the VISA address
- Click Connect
- Verify connection with the Test button

### 2. Channel Configuration

Configure the channels you want to use:
- Select active channels (Ch1-Ch4)
- Set voltage scale (V/div)
- Configure offset, coupling (DC/AC), and probe attenuation
- Apply the configuration

Optional: Use Autoscale for automatic channel setup

### 3. Timebase and Trigger Settings

**Trigger Configuration:**
- Source: Select trigger channel
- Level: Set voltage threshold
- Slope: Rising, Falling, or Either

**Timebase:**
- Set time/division to display 2-3 signal cycles

### 4. Capture Parameters

**Acquisition Settings:**
- Number of captures (1-10,000)
- Interval between captures (seconds)
- Trigger timeout (seconds)

**Data Selection:**
- Choose channels to save
- Enable/disable screenshot capture
- Enable/disable waveform data export
- Optional: Enable combined multi-channel CSV

**File Settings:**
- Base filename prefix
- Save directory path

### 5. Running a Capture Session

1. Navigate to Capture Control tab
2. Click Start Capture
3. Monitor progress using Refresh or enable Auto Refresh
4. Stop early if needed using the Stop button
5. Review results in the Results tab

---

## Output Files

### File Naming Convention

**Screenshots:**
```
<prefix>_screenshot_<index>_<YYYYMMDD>_<HHMMSS>_<ms>.png
```

**Waveform Data:**
```
<prefix>_CH<n>_<index>_<YYYYMMDD>_<HHMMSS>_<ms>.csv
```

**Combined Multi-Channel:**
```
<prefix>_MULTI_<index>_<YYYYMMDD>_<HHMMSS>_<ms>.csv
```

### File Formats

**PNG Screenshots:**
- Oscilloscope display capture at trigger event

**CSV Waveform Files:**
- Header: Channel info, timestamp, sample rate, point count
- Data columns: Time (s), Voltage (V)
- Example:
```csv
# Channel: 1
# Capture Index: 0
# Timestamp: 2025-01-22T14:30:52.123456
# Sample Rate: 5.00e+09 Hz
# Points: 62500

Time (s),Voltage (V)
-0.000001000,0.0234
-0.000000998,0.0245
...
```

**Combined CSV Format:**
- All channels time-aligned in single file
- Columns: Time (s), CH1 (V), CH2 (V), etc.

### Data Analysis

Import captured data using:
- **Excel**: Import CSV with comma delimiter
- **MATLAB**: `data = readtable('filename.csv')`
- **Python**: `df = pd.read_csv('filename.csv', comment='#')`

---

## Best Practices

### Recommended Workflow

1. Verify trigger stability on the oscilloscope before starting automated capture
2. Set trigger timeout with appropriate margin (typically 2-10x signal period)
3. Start with a test run (5-10 captures) to verify configuration
4. Check initial output files for correctness
5. Monitor capture progress using Auto Refresh
6. Address failures promptly if success rate drops below 95%

### Common Pitfalls

- Setting interval too short for slow signals causes repeated timeouts
- Large capture sessions without preliminary testing
- Closing browser during active capture prevents status monitoring
- Insufficient trigger timeout for intermittent signals

---

## Troubleshooting

### Connection Failures

**Symptoms:** Connection test fails or oscilloscope not detected

**Solutions:**
- Verify VISA address is correct
- Confirm oscilloscope is powered on and USB/LAN is connected
- Install NI-VISA drivers if not present
- Check cable connections

### Repeated Trigger Timeouts

**Symptoms:** All or most captures timing out

**Causes:**
- No signal present on trigger channel
- Incorrect trigger level or slope
- Timeout value too short for signal period

**Solutions:**
- Verify signal is visible on oscilloscope display
- Adjust trigger level to ~50% of signal amplitude
- Increase trigger timeout for slow or intermittent signals
- Change trigger slope to "Either" if unsure of edge direction

### File I/O Issues

**Symptoms:** Empty or corrupted files, write errors

**Solutions:**
- Verify sufficient disk space (~10 MB per 100 captures)
- Check write permissions for save directory
- Ensure parent directory exists
- Test with a simple path (e.g., Desktop)

### Performance Issues

**Expected Performance:** 1-2 seconds per capture is typical

**Optimization:**
- Use USB connection instead of LAN
- Disable unused channels
- Reduce interval between captures
- Note: High-resolution waveforms (62,500 points) inherently take time to transfer

### Display Capture Problems

**Symptoms:** Blank or black screenshots

**Solutions:**
- Verify oscilloscope display is active
- Run Autoscale to ensure waveforms are visible
- Check display settings on oscilloscope

---

## System Architecture

### Capture Loop Operation

1. Set oscilloscope to SINGLE trigger mode
2. Wait for trigger event (up to timeout duration)
3. On trigger:
   - Capture display as PNG
   - Download waveform data
   - Save files with timestamp
4. Wait for specified interval
5. Repeat until capture count reached

### Timing Characteristics

**Total Duration ≈ N × (Interval + Capture_Time)**

Where:
- N = Number of captures
- Interval = User-defined wait time
- Capture_Time ≈ 1-2 seconds (data transfer overhead)

Example: 50 captures with 1.0s interval ≈ 125 seconds total

Note: Interval represents target loop time. Actual wait time is adjusted to compensate for capture overhead.

---

## Application Examples

### Pulsed Signal Characterization

**Scenario:** Capturing laser photodiode pulses (2 Hz repetition rate)

**Configuration:**
- Captures: 100
- Interval: 2.0s
- Timeout: 5.0s
- Trigger: CH1, Rising edge, 0.5V threshold

**Output:** 100 synchronized screenshots and waveform files

### Transient Event Monitoring

**Scenario:** Power supply glitch detection

**Configuration:**
- Captures: 50
- Interval: 0s (continuous)
- Timeout: 60s
- Trigger: CH2, Rising edge, 5.5V threshold (above nominal 5V)

**Use:** Automatic capture of infrequent voltage transients

### Multi-Channel Analysis

**Scenario:** Input-output phase relationship verification

**Configuration:**
- Captures: 200
- Interval: 5.0s
- Channels: CH1, CH2
- Trigger: CH1
- Combined CSV: Enabled

**Output:** Time-aligned multi-channel data for correlation analysis

---

## Parameter Reference

### Trigger Timeout

Maximum wait time for trigger event before aborting capture.

**Recommendations:**
- Fast signals (<100ms period): 1-2s
- Moderate signals (0.1-1s period): 5-10s
- Slow/intermittent signals: 30-120s
- Rule of thumb: Set to 2-10x expected signal period

### Interval Between Captures

Target time between consecutive capture starts.

**Behavior:**
- Actual wait = Interval - Capture_Time
- When Interval = 0: Back-to-back captures at maximum hardware speed (~0.5-1 Hz)
- For periodic signals: Match or exceed signal period

### Combined Multi-Channel CSV

Saves all channels in single time-aligned file.

**Use when:**
- Performing cross-channel analysis
- Need guaranteed time alignment
- Prefer simplified file management

**Avoid when:**
- Single channel acquisition
- Concerned about file size
- Prefer modular per-channel files

---

## Data Management

### Recommended Directory Structure

```
OscilloscopeData/
├── 2025-01-22_LaserTest/
│   ├── trigger_capture_screenshot_*.png
│   ├── trigger_capture_CH1_*.csv
│   └── capture_summary.json
├── 2025-01-23_PowerSupplyTest/
└── 2025-01-24_SignalAnalysis/
```

Use dated folders with descriptive names for each test session.

### Configuration Presets

| Application | Captures | Interval | Timeout |
|-------------|----------|----------|---------|
| Fast repetitive signal | 50-100 | 0.5s | 2s |
| 1 Hz periodic signal | 100-200 | 1.0s | 5s |
| Rare transient events | 20-50 | 0s | 60-120s |
| Long-term monitoring | 500-1000 | 60s | 120s |
| Configuration test | 5-10 | 1s | 10s |

---

## Pre-Capture Checklist

- [ ] Oscilloscope powered and connected
- [ ] Signal visible on display
- [ ] Stable trigger (non-flickering waveform)
- [ ] Channels configured correctly
- [ ] Trigger level set appropriately
- [ ] Timebase displays 2-3 signal cycles
- [ ] Save directory path verified
- [ ] Initial test run completed
- [ ] Sample files verified

---

## Error Reference

| Error Message | Cause | Resolution |
|---------------|-------|------------|
| "Not connected" | No oscilloscope connection | Verify connection in Connection tab |
| "No channels selected" | No channels enabled for data capture | Enable at least one channel |
| "Configuration error" | Invalid parameter values | Check all parameters are valid |
| "Trigger timeout" | No trigger event detected | Verify signal and trigger settings |

---

## Technical Support

**Developer**: Anirudh Iyengar
**Organization**: Digantara Research and Technologies Pvt. Ltd.

For assistance, contact the instrumentation team.

---

## License

Internal use - Digantara Research and Technologies Pvt. Ltd.
