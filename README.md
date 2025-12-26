# 📸 Continuous Trigger Capture System

## Automated Screenshot & Waveform Data Capture for Keysight Oscilloscope

**Developer**: Anirudh Iyengar
**Organization**: Digantara Research and Technologies Pvt. Ltd.
**Version**: 2.0.0
**Date**: January 22, 2025

---

## 🎯 What This Tool Does

This is a simple, automated system that captures oscilloscope screenshots and waveform data **every time a trigger event occurs**. Think of it like a camera that automatically takes a photo of your oscilloscope screen whenever it detects a signal.

### In Simple Terms:

✅ **Waits for a signal** (trigger event)
✅ **Captures the screen** as a PNG image
✅ **Saves the waveform data** as CSV files
✅ **Repeats** for as many captures as you want
✅ **Shows progress** in real-time
✅ **Organizes all files** neatly in folders

**Perfect for**: Collecting multiple signal measurements automatically without sitting at the oscilloscope clicking "Save" repeatedly!

---

## 🚀 Quick Start Guide (3 Simple Steps)

### Step 1: Install Required Software

Open a command prompt and type:

```bash
pip install -r requirements.txt
```

This installs all the needed libraries (numpy, pandas, gradio, etc.)

### Step 2: Launch the Application

```bash
python continuous_trigger_capture.py
```

The program will start and automatically open in your web browser!

### Step 3: Use the Web Interface

1. **Connect to your oscilloscope** (Connection tab)
2. **Set up your signal** (Channel Configuration tab)
3. **Configure trigger** (Timebase & Trigger tab)
4. **Start capturing** (Capture Setup → Capture Control)

That's it! The system does the rest automatically.

---

## 📋 Complete Step-by-Step Instructions

### STEP 1: Connect to Your Oscilloscope

1. **Find your oscilloscope's address**:
   - Look at the oscilloscope screen
   - Press **Utility** button → **I/O** → **LAN Configuration**
   - Note down the VISA address (looks like: `USB0::0x0957::0x1780::MY65220169::INSTR`)

2. **In the web interface**:
   - Go to **"Connection"** tab
   - Paste the VISA address in the textbox
   - Click **"Connect"** button
   - Wait for "Connected to Keysight..." message

3. **Test the connection**:
   - Click **"Test"** button
   - Should say "Connection test: PASSED"

✅ **You're now connected!**

---

### STEP 2: Configure Your Channels

1. **Go to "Channel Configuration" tab**

2. **Select which channels to use**:
   - Check ☑ **Ch1** if using Channel 1
   - Check ☑ **Ch2** if using Channel 2
   - Check ☑ **Ch3** if using Channel 3
   - Check ☑ **Ch4** if using Channel 4

3. **Set voltage scale** (V/div):
   - Example: `1.0` means 1 volt per division
   - Adjust so your signal fills about 50-80% of screen height

4. **Set other parameters**:
   - **Offset**: Usually `0.0` (centers the signal)
   - **Coupling**: Choose `DC` (most common) or `AC`
   - **Probe**: Select `10x` if using 10:1 probe, `1x` otherwise

5. **Click "Configure Channels"**

6. **Optional - Auto Setup**:
   - Click **"Autoscale"** to let the scope automatically adjust settings
   - Wait 3 seconds for it to finish

✅ **Your channels are now set up!**

---

### STEP 3: Set Up Timebase & Trigger

1. **Go to "Timebase & Trigger" tab**

#### Configure Trigger:

**What is a trigger?** It tells the scope when to capture a signal (e.g., when voltage rises above 1V)

- **Trigger Source**: Choose which channel to trigger on (usually `CH1`)
- **Trigger Level**: Voltage threshold (e.g., `0.0` for zero crossing, `1.5` for 1.5V)
- **Trigger Slope**:
  - `Rising` = trigger when voltage goes up ↗
  - `Falling` = trigger when voltage goes down ↘
  - `Either` = trigger on both
- Click **"Apply Trigger"**

#### Set Timebase:

**What is timebase?** How much time each horizontal division represents

- **Time/div**: Select from dropdown (e.g., `1 ms` = 1 millisecond per division)
- Adjust so you see 2-3 complete signal cycles on screen
- Click **"Apply Timebase"**

✅ **Trigger and timebase configured!**

---

### STEP 4: Configure Capture Settings

1. **Go to "Capture Setup" tab**

#### Capture Parameters:

- **Number of Captures**: How many screenshots/waveforms to collect
  - Example: `50` = capture 50 trigger events
  - Range: 1 to 10,000

- **Interval Between Captures**: Time to wait between each capture (in seconds)
  - Example: `1.0` = wait 1 second between captures
  - Use `0` for back-to-back captures (as fast as possible)
  - Use `5.0` for slower signals (wait 5 seconds between)

- **Trigger Timeout**: Maximum time to wait for a trigger (in seconds)
  - Example: `10.0` = wait up to 10 seconds for signal
  - Increase if your signal is slow/intermittent
  - Decrease for faster operation

#### Channel Selection:

Check which channels to save data from:
- ☑ **Channel 1**
- ☑ **Channel 2**
- ☐ **Channel 3**
- ☐ **Channel 4**

#### Save Options:

Choose what to save:

- ☑ **Capture Screenshots**: Saves PNG image of oscilloscope screen
  - Recommended: YES (to see what the signal looked like)

- ☑ **Save Waveform Data**: Saves CSV files with time/voltage data
  - Recommended: YES (for later analysis in Excel/MATLAB)

- ☐ **Save Combined Multi-Channel CSV**: Saves all channels in one CSV file
  - Optional: Check if you want all channels together

#### File Settings:

- **Base Filename**: Name prefix for all files
  - Example: `trigger_capture` → files named `trigger_capture_0001.png`, etc.

- **Save Directory**: Where to save files
  - Default: `trigger_captures` folder in current directory
  - Can change to any folder path: `C:\Users\YourName\Documents\MyData`

#### Time Estimate:

The system shows estimated total time based on your settings.

✅ **Capture configured!**

---

### STEP 5: Start Capturing!

1. **Go to "Capture Control" tab**

2. **Click the big "Start Capture" button**

3. **Monitor progress**:
   - Click **"Refresh"** to update status
   - OR check **"Auto Refresh (2s)"** for automatic updates every 2 seconds

4. **Watch the status display**:
   ```
   RUNNING
   -----------------------------------------
   Progress: 15/50 (30.0%)
   [██████░░░░░░░░░░░░░░]
   Successful: 14
   Failed: 1
   Total Files: 42
   ```

5. **To stop early**: Click **"Stop"** button

6. **Wait for completion**: Status will show "IDLE" when done

✅ **Capturing in progress!**

---

### STEP 6: View Your Results

1. **Go to "Results" tab**

2. **Click "Show Files"** to see list of captured files:
   ```
   CAPTURED FILES
   -----------------------------------------

   Screenshots (50):
     • trigger_capture_screenshot_0001_20250122_143052_123.png
     • trigger_capture_screenshot_0002_20250122_143053_456.png
     ...

   Waveforms (100):
     • trigger_capture_CH1_0001_20250122_143052_123.csv
     • trigger_capture_CH2_0001_20250122_143052_123.csv
     ...
   ```

3. **Click "Generate Summary"** for detailed statistics:
   ```
   CAPTURE SESSION SUMMARY
   -----------------------------------------

   STATISTICS:
     • Total Captures: 50
     • Successful: 48 (96.0%)
     • Failed: 2
     • Success Rate: 96.0%

   TIMING:
     • Total Duration: 62.3 seconds
     • Avg Time/Capture: 1.30 seconds

   FILES SAVED:
     • Screenshots: 48
     • Waveform Files: 96
     • Total Files: 144
   ```

4. **Click "Save Report"** to export summary as JSON file

5. **Find your files**:
   - Open the **Save Directory** you specified
   - All PNG and CSV files are there, ready to use!

✅ **Results saved and ready!**

---

## 📁 Understanding Your Files

After a capture session, you'll have these files:

### Screenshot Files (PNG):
```
trigger_capture_screenshot_0001_20250122_143052_123.png
                          ↑       ↑        ↑      ↑
                       index    date     time   ms
```
- **Index**: Capture number (0001, 0002, 0003...)
- **Date**: YYYYMMDD format
- **Time**: HHMMSS format
- **Milliseconds**: For precise timing

**What's inside?** A picture of the oscilloscope screen, exactly as it appeared when the trigger fired.

### Waveform CSV Files:
```
trigger_capture_CH1_0001_20250122_143052_123.csv
```

**What's inside?** Numerical data with columns:
- `Time (s)`: Time values (in seconds)
- `Voltage (V)`: Voltage values (in volts)

**Example CSV content:**
```csv
# Channel: 1
# Capture Index: 0
# Timestamp: 2025-01-22T14:30:52.123456
# Sample Rate: 5.00e+09 Hz
# Points: 62500

Time (s),Voltage (V)
-0.000001000,0.0234
-0.000000998,0.0245
-0.000000996,0.0251
...
```

**How to use:**
- Open in **Excel**: Import as CSV file
- Open in **MATLAB**: `data = readtable('filename.csv')`
- Open in **Python**: `df = pd.read_csv('filename.csv', comment='#')`

### Combined CSV Files (if enabled):
```
trigger_capture_MULTI_0001_20250122_143052_123.csv
```

**What's inside?** All channels in one file:
```csv
Time (s),CH1 (V),CH2 (V)
-0.000001000,0.0234,0.1456
-0.000000998,0.0245,0.1467
...
```

---

## 💡 Tips for Best Results

### ✅ DO:

1. **Test your trigger first**:
   - On the oscilloscope, make sure the trigger is working
   - Signal should be stable, not flickering
   - See a steady waveform on screen

2. **Use appropriate timeout**:
   - If signal repeats every 0.1s → use 1s timeout (10× safety margin)
   - If signal is slow (every 5s) → use 20s timeout

3. **Start with a small test**:
   - Try 5 captures first to verify everything works
   - Then increase to 50 or 100 for real data collection

4. **Check the first few files**:
   - After starting, click "Show Files" after 30 seconds
   - Open one PNG to verify it looks correct
   - Open one CSV to verify data looks reasonable

5. **Monitor the capture**:
   - Enable "Auto Refresh" in Capture Control tab
   - Watch for failures (should be 0 or very few)
   - If many failures, stop and fix trigger settings

### ❌ DON'T:

1. **Don't use interval = 0 for slow signals**:
   - System will timeout repeatedly if signal doesn't come fast enough
   - Use interval ≥ your signal period

2. **Don't start huge captures without testing**:
   - Test with 5 captures first
   - Verify files are created and look correct
   - Then scale up to hundreds/thousands

3. **Don't ignore failures**:
   - If "Failed: 10" out of 50, something is wrong
   - Usually means: bad trigger, wrong timeout, or signal issues

4. **Don't close the browser window during capture**:
   - Capture runs in background, but you won't see status
   - Keep the window open to monitor progress

---

## 🔍 Troubleshooting Common Issues

### Problem: "Connection test: FAILED"

**Causes:**
- Wrong VISA address
- Oscilloscope is off
- USB cable disconnected
- VISA drivers not installed

**Solutions:**
1. Check oscilloscope is powered on
2. Verify USB cable is connected
3. Install NI-VISA drivers: https://www.ni.com/en-us/support/downloads/drivers/download.ni-visa.html
4. Try different VISA address format

---

### Problem: "Trigger timeout" on every capture

**Causes:**
- No signal on trigger channel
- Trigger level wrong
- Trigger timeout too short
- Signal not repeating

**Solutions:**
1. Look at oscilloscope screen - do you see a stable waveform?
2. If no signal: check cables, signal source
3. If signal visible but not triggering:
   - Adjust trigger level (try 50% of signal amplitude)
   - Change trigger slope (try "Either")
4. If signal is slow: increase Trigger Timeout to 30 or 60 seconds

---

### Problem: Files are created but are empty or corrupted

**Causes:**
- Disk full
- No write permissions
- Path doesn't exist

**Solutions:**
1. Check disk space (need ~10 MB per 100 captures)
2. Verify Save Directory path is correct
3. Try saving to Desktop first: `C:\Users\YourName\Desktop\test_data`
4. Make sure folder path exists (system creates it, but parent must exist)

---

### Problem: Capture is very slow

**Causes:**
- Large interval between captures
- Slow network (if using LAN connection)
- Many channels enabled
- Large waveform point count (62,500 points per channel)

**Solutions:**
1. Reduce Interval Between Captures
2. Use USB connection instead of LAN (much faster)
3. Disable unused channels
4. This is normal for high-resolution captures (1-2 seconds per capture is typical)

---

### Problem: Screenshot is blank/black

**Causes:**
- Scope display is off
- Wrong display mode

**Solutions:**
1. Press the oscilloscope's **Display** button
2. Make sure screen is on and showing waveforms
3. Try clicking "Autoscale" first

---

### Problem: CSV files can't be opened

**Causes:**
- File still being written (capture in progress)
- Associated program doesn't support CSV

**Solutions:**
1. Wait for capture to complete before opening files
2. Right-click file → "Open with" → Excel or Notepad
3. Files have header comments starting with `#` - some programs may need these skipped

---

## 🎓 Understanding the System

### How It Works (Simple Explanation):

1. **SINGLE Mode**: System puts oscilloscope in "SINGLE" mode
   - This means: capture ONE trigger event, then stop

2. **Wait for Trigger**: System waits up to `Trigger Timeout` seconds
   - Watching for your signal to cross the trigger level

3. **Trigger Fires**: When signal crosses threshold:
   - Oscilloscope freezes the display
   - System captures the screen as PNG
   - System downloads waveform data as CSV

4. **Save Files**: All files saved with timestamp

5. **Wait**: System waits `Interval Between Captures` seconds

6. **Repeat**: Go back to step 1, until `Number of Captures` reached

### Timing Explanation:

**Total Time ≈ (Number of Captures) × (Interval + Capture Time)**

Example:
- Number of Captures = 50
- Interval = 1.0 second
- Capture Time ≈ 1-2 seconds (screenshot + data download)
- **Total Time ≈ 50 × (1.0 + 1.5) = 125 seconds ≈ 2 minutes**

The "Interval Between Captures" is the **target total loop time**, not just the wait time. The system adjusts the actual wait to account for capture time.

---

## 📊 Example Use Cases

### Use Case 1: Characterizing a Pulsed Laser

**Setup:**
- Connect laser photodiode to CH1
- Laser fires every 2 seconds
- Want 100 pulse waveforms

**Configuration:**
```
Number of Captures: 100
Interval: 2.0 seconds
Trigger Timeout: 5.0 seconds (2× safety)
Channel: CH1
Trigger Source: CH1
Trigger Level: 0.5 V (50% of pulse height)
Trigger Slope: Rising
Capture Screenshots: Yes
Save Waveforms: Yes
```

**Result:** 100 PNG screenshots + 100 CSV files of laser pulses

---

### Use Case 2: Capturing Transient Events

**Setup:**
- Monitoring power supply voltage on CH2
- Want to capture voltage glitches whenever they occur
- Glitches happen randomly every 10-30 seconds

**Configuration:**
```
Number of Captures: 50
Interval: 0.0 seconds (capture as fast as they come)
Trigger Timeout: 60.0 seconds (glitches are rare)
Channel: CH2
Trigger Source: CH2
Trigger Level: 5.5 V (above normal 5V level)
Trigger Slope: Rising
Capture Screenshots: Yes
Save Waveforms: Yes
```

**Result:** Every time voltage exceeds 5.5V, capture it automatically

---

### Use Case 3: Comparing Two Signals Over Time

**Setup:**
- Input signal on CH1
- Output signal on CH2
- Want to verify phase relationship stays constant

**Configuration:**
```
Number of Captures: 200
Interval: 5.0 seconds
Trigger Timeout: 10.0 seconds
Channels: CH1, CH2
Trigger Source: CH1
Save Combined CSV: Yes
```

**Result:** 200 combined CSV files with both channels for offline analysis

---

## 🔧 Advanced Settings Explained

### Trigger Timeout:

**What it does:** Maximum time to wait for a trigger before giving up

**When to increase:**
- Slow/intermittent signals
- Signal period > 1 second
- Capturing rare events

**When to decrease:**
- Fast signals (< 100 ms period)
- Want faster error detection
- Signal should always be present

---

### Interval Between Captures:

**What it does:** Target time from start of one capture to start of next

**Formula:** `Actual wait = Interval - (time to capture + time to save)`

**Special case: Interval = 0**
- No extra wait
- Captures back-to-back as fast as possible
- Actual rate limited by oscilloscope speed (~1-2 captures/second)

---

### Save Combined Multi-Channel CSV:

**When to use:**
- Want all channels time-aligned in one file
- Easier for plotting (one file instead of multiple)
- Doing correlation analysis between channels

**When NOT to use:**
- Only using one channel (no benefit)
- Want separate files for each channel (easier to manage)
- File size concerns (combined files are larger)

---

## 📚 File Organization Best Practices

### Recommended Folder Structure:

```
C:\Users\YourName\Documents\OscilloscopeData\
├── 2025-01-22_Laser_Test\
│   ├── trigger_capture_screenshot_0001.png
│   ├── trigger_capture_screenshot_0002.png
│   ├── trigger_capture_CH1_0001.csv
│   ├── trigger_capture_CH1_0002.csv
│   └── capture_summary_20250122_143052.json
│
├── 2025-01-23_Power_Supply_Test\
│   └── ...
│
└── 2025-01-24_Signal_Analysis\
    └── ...
```

**Tips:**
- Create a new folder for each test/experiment
- Use descriptive names with dates
- Keep related captures together
- Back up important data regularly

---

## 🎯 Quick Reference Card

### Typical Settings for Common Scenarios:

| Scenario | Captures | Interval | Timeout | Notes |
|----------|----------|----------|---------|-------|
| **Fast repetitive signal** | 50 | 0.5 s | 2 s | Quick collection |
| **Slow signal (1 Hz)** | 100 | 1.0 s | 5 s | Match signal rate |
| **Rare events** | 50 | 0 s | 60 s | Wait patiently |
| **Long-term monitoring** | 1000 | 60 s | 120 s | Every minute for hours |
| **Quick test** | 5 | 1 s | 10 s | Verify setup |

---

## ✅ Pre-Flight Checklist

Before starting a capture session:

- [ ] Oscilloscope is on and connected
- [ ] Signal is visible on screen
- [ ] Trigger is working (stable waveform, not flickering)
- [ ] Channels are enabled and configured
- [ ] Trigger level is appropriate (usually 50% of signal)
- [ ] Timebase shows 2-3 signal cycles
- [ ] Save Directory path is correct
- [ ] Tested with 5 captures first
- [ ] Checked that files are being created
- [ ] Verified first screenshot looks correct

**If all checked → Ready to start full capture!**

---

## 🆘 Getting Help

### Still Having Problems?

1. **Check the logs**: Look at the console window for error messages
2. **Test manually**: Try saving one screenshot manually on the scope
3. **Simplify**: Try 1 channel, 5 captures, basic settings
4. **Verify connection**: Use oscilloscope's built-in web server to confirm network works

### Common Error Messages:

| Error | Meaning | Fix |
|-------|---------|-----|
| "Not connected" | No oscilloscope connection | Go to Connection tab, click Connect |
| "No channels selected" | All channel checkboxes unchecked | Check at least one channel |
| "Configuration error" | Invalid settings | Check all numbers are positive |
| "Trigger timeout" | No signal detected | Check signal, adjust trigger level |

---

## 📖 Summary

### What You've Learned:

✅ How to connect to the oscilloscope
✅ How to configure channels and trigger
✅ How to set up and start captures
✅ How to monitor progress
✅ How to find and use your captured files
✅ How to troubleshoot common issues

### Key Takeaways:

1. **Test first** - Always start with 5 captures to verify setup
2. **Monitor progress** - Use Auto Refresh to watch status
3. **Check files early** - Open first PNG/CSV to verify correctness
4. **Adjust as needed** - If failures occur, stop and fix settings
5. **Save your data** - Organize in dated folders with good names

---

## 🎉 You're Ready!

This tool will save you hours of manual work by automating repetitive oscilloscope captures.

**Start small, verify it works, then scale up to hundreds or thousands of automated captures!**

---

## 📞 Support

**Developed by**: Anirudh Iyengar
**Organization**: Digantara Research and Technologies Pvt. Ltd.

For technical support, contact your organization's instrumentation team.

---

## 🚀 Quick Start Reminder

```bash
# 1. Install
pip install -r requirements.txt

# 2. Launch
python continuous_trigger_capture.py

# 3. In web browser:
#    - Connect (Connection tab)
#    - Configure (Channel & Trigger tabs)
#    - Capture (Capture Setup → Control)
#    - Results (Results tab)
```

**Good luck with your measurements! 📊🔬**
