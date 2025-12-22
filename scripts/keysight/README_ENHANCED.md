# 🔬 Laser-APD Delay Analysis System v3.0

## Professional Waveform Capture & Timing Jitter Measurement

**Developer**: Anirudh Iyengar
**Organization**: Digantara Research and Technologies Pvt. Ltd.
**Version**: 3.0.0 - Enhanced Edition
**Date**: January 22, 2025

---

## 🎯 What This Does

Automatically captures oscilloscope waveforms from Laser (CH1) and APD (CH2) signals, then:

✅ **Measures delay** between signals using cross-correlation
✅ **Calculates jitter** statistics in real-time
✅ **Generates plots** showing delay distribution
✅ **Exports data** to Excel, MATLAB, CSV formats
✅ **Provides professional reports** with statistical analysis

**Perfect for**: Studying timing stability, characterizing signal delays, measuring jitter in optical systems

---

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies

```bash
pip install -r requirements_enhanced.txt
```

### Step 2: Launch Application

```bash
python continuous_trigger_capture_enhanced.py
```

### Step 3: Use Quick Start Tab

1. Connect to oscilloscope
2. Go to "🚀 Quick Start" tab
3. Set captures (e.g., 50) and interval (e.g., 1s)
4. Click "START"
5. View live results in "📊 Live Analysis"

**That's it!** System will automatically:
- Trigger on each signal
- Capture waveforms
- Calculate delays
- Update statistics live
- Save all data

---

## 📋 Key Features

### 🎯 CRITICAL Features (v3.0)

| Feature | Description | Benefit |
|---------|-------------|---------|
| **Auto Delay Calculation** | Cross-correlation algorithm | No manual analysis needed |
| **Real-Time Statistics** | Live mean, σ, jitter updates | Monitor quality during capture |
| **Live Plots** | Histogram + time series | Visual feedback during acquisition |
| **Statistical Reports** | Comprehensive analysis | Publication-ready results |
| **Multi-Format Export** | Excel, MATLAB, CSV, PNG | Works with your tools |
| **Quick Start UI** | One-tab setup | Easy for new users |

### 📊 What You Get

**After 50 captures, you'll have:**

```
laser_apd_delay_20250122_143052/
├── screenshots/              # 50 PNG images of scope display
├── waveforms/                # 100 CSV files (2 channels × 50)
└── analysis/
    ├── analysis.xlsx         # Excel with stats + raw data
    ├── analysis.mat          # MATLAB format
    ├── delay_plot.png        # Publication figure
    └── report.txt            # Statistical summary
```

**Report includes:**
- Mean delay ± uncertainty
- RMS jitter
- Peak-to-peak jitter
- 95% confidence interval
- Distribution analysis
- Signal quality metrics

---

## 📊 Example Results

```
╔════════════════════════════════════════════════════════════════╗
║        SIGNAL DELAY ANALYSIS REPORT                            ║
╚════════════════════════════════════════════════════════════════╝

DELAY STATISTICS (Laser → APD)
────────────────────────────────────────────────────────────────
Mean Delay:                152.345 ns
Standard Deviation (σ):    3.214 ns
Median Delay:              152.123 ns

JITTER ANALYSIS
────────────────────────────────────────────────────────────────
RMS Jitter (1σ):           3.214 ns
Peak-to-Peak Jitter:       14.198 ns

CONFIDENCE INTERVAL
────────────────────────────────────────────────────────────────
95% CI:                    152.345 ± 0.890 ns
```

**Interpretation**:
- Your laser-APD delay is **152.3 ± 0.9 ns** (95% confidence)
- Timing jitter is **3.2 ns RMS**
- This means 95% of measurements fall within ±6.4 ns

---

## 🔬 Scientific Methodology

### Cross-Correlation Algorithm

**How it works:**
1. Normalize both signals (remove DC, scale to unit variance)
2. Compute cross-correlation using FFT
3. Find peak correlation → that's the delay
4. Calculate confidence based on peak sharpness

**Accuracy:**
- **Resolution**: 200 ps (at 5 GSa/s sample rate)
- **Precision**: ±0.5 ns (typical, with good SNR)
- **Jitter floor**: < 100 ps RMS (ideal conditions)

**Advantages over manual methods:**
- Sub-sample precision through interpolation
- Robust to noise
- Automatic, no user bias
- Fast (uses FFT)

---

## 📖 Documentation

### Files in This Package

| File | Purpose |
|------|---------|
| `continuous_trigger_capture_enhanced.py` | **Main application** (v3.0 - Enhanced) |
| `continuous_trigger_capture.py` | Original version (v2.0 - Basic) |
| `ENHANCED_VERSION_GUIDE.md` | **Complete user manual** (60+ pages) |
| `INSTALL_INSTRUCTIONS.md` | Installation help |
| `requirements_enhanced.txt` | Python dependencies |
| `README_ENHANCED.md` | This file |

### Where to Start

1. **New user?** → Read [INSTALL_INSTRUCTIONS.md](INSTALL_INSTRUCTIONS.md)
2. **Want details?** → Read [ENHANCED_VERSION_GUIDE.md](ENHANCED_VERSION_GUIDE.md)
3. **Just want to run?** → Follow Quick Start above

---

## 🎮 User Interface Tabs

### 🚀 Quick Start (RECOMMENDED)
- **Simplest way to begin**
- Essential settings only
- One-click start
- Perfect for routine measurements

### 🔌 Connection
- Connect to oscilloscope
- Test connection
- View instrument info

### ⚙️ Channel Configuration
- Set voltage scales
- Configure probe attenuation
- Enable/disable channels
- Autoscale function

### 🎯 Timebase & Trigger
- Set timebase (time/div)
- Configure trigger level
- Choose trigger slope

### 📋 Advanced Capture Setup
- Fine-grained control
- Custom delay analysis settings
- All save options
- For power users

### 🎮 Capture Control
- Start/stop capture
- Monitor progress
- View live status
- ETA calculation

### 📊 Live Analysis ⭐ NEW
- **Real-time delay plots**
- **Live statistics updates**
- Auto-refresh every 3s
- See results as they come in

### 📈 Results & Reports
- View captured files
- Generate statistical report
- Export all formats
- Professional output

---

## 🔧 System Requirements

### Minimum
- Python 3.8+
- 4 GB RAM
- Windows 10 / Linux / macOS
- Keysight DSOX6004A oscilloscope

### Recommended
- Python 3.11+
- 8 GB RAM
- Windows 11 / Ubuntu 22.04
- Fast SSD for data storage

### Dependencies
- numpy (numerical computing)
- pandas (data handling)
- matplotlib (plotting)
- scipy (cross-correlation)
- gradio (web UI)
- openpyxl (Excel export)
- pyvisa (scope communication)

---

## 💡 Best Practices

### For Accurate Measurements

✅ **DO:**
- Use ≥50 captures for statistics
- Verify stable triggering
- Check confidence scores (aim for > 0.7)
- Shield cables from noise
- Monitor live statistics

❌ **DON'T:**
- Use < 30 captures (poor statistics)
- Ignore low confidence warnings
- Let signals clip (saturate)
- Skip verification plots
- Assume all captures are good

### Signal Quality Checklist

Before starting capture:
- [ ] Signals visible on screen
- [ ] No clipping (flat tops)
- [ ] Stable triggering (not flickering)
- [ ] Trigger level at 50% amplitude
- [ ] Timebase shows 2-3 signal periods
- [ ] Both channels enabled

---

## 📊 What the Statistics Mean

| Metric | What It Tells You | Good Value |
|--------|-------------------|------------|
| **Mean Delay** | Average time offset | Depends on setup |
| **RMS Jitter** | Timing noise (1σ) | < 5 ns (good) |
| **Pk-Pk Jitter** | Worst-case variation | < 20 ns (good) |
| **Confidence** | Measurement quality | > 0.7 (acceptable) |
| **CV %** | Relative precision | < 5% (good) |
| **95% CI** | Uncertainty in mean | Smaller = better |

### Jitter Interpretation

- **< 1 ns RMS**: Excellent (low-jitter system)
- **1-5 ns RMS**: Good (typical for optical systems)
- **5-10 ns RMS**: Fair (may need improvement)
- **> 10 ns RMS**: Poor (investigate noise sources)

---

## 🔍 Troubleshooting Common Issues

### "Trigger Timeout" Errors
- **Cause**: No signal on trigger channel
- **Fix**: Check cables, adjust trigger level, use autoscale

### Low Confidence Scores (< 0.5)
- **Cause**: Poor SNR or wrong channels
- **Fix**: Increase signal amplitude, check channel assignment, reduce noise

### Unrealistic Delays
- **Cause**: Channels swapped or period ambiguity
- **Fix**: Verify CH1 = Laser, CH2 = APD; reduce timebase

### Slow Acquisition
- **Cause**: Long trigger timeout or network delays
- **Fix**: Reduce timeout to 5s, use USB not LAN

See [ENHANCED_VERSION_GUIDE.md](ENHANCED_VERSION_GUIDE.md) for complete troubleshooting guide.

---

## 🆚 Version Comparison

### v3.0 Enhanced vs. v2.0 Basic

| Feature | v2.0 Basic | v3.0 Enhanced |
|---------|-----------|---------------|
| Waveform capture | ✅ | ✅ |
| Screenshots | ✅ | ✅ |
| CSV export | ✅ | ✅ |
| **Auto delay calc** | ❌ | ✅ NEW |
| **Real-time stats** | ❌ | ✅ NEW |
| **Live plots** | ❌ | ✅ NEW |
| **Excel export** | ❌ | ✅ NEW |
| **MATLAB export** | ❌ | ✅ NEW |
| **Statistical reports** | ❌ | ✅ NEW |
| **Quick Start UI** | ❌ | ✅ NEW |
| **Confidence scoring** | ❌ | ✅ NEW |

**Recommendation**: Use v3.0 Enhanced for all new work. v2.0 remains available for compatibility.

---

## 📚 Example Workflows

### Workflow 1: Quick Delay Measurement (15 minutes)

```
1. Connect scope → "Connection" tab
2. Autoscale → "Channel Configuration" tab
3. Quick Start → Set 50 captures, 1s interval
4. Click START
5. Wait 15 minutes (50 × 1s + overhead)
6. View results → "Live Analysis" tab
7. Export → "Results & Reports" tab
```

**Output**: Mean delay ± uncertainty, jitter statistics, plots

### Workflow 2: High-Precision Characterization (2 hours)

```
1. Connect and configure scope
2. Advanced Setup → 200 captures, 2s interval
3. Enable all save options
4. Monitor live statistics during acquisition
5. Generate comprehensive report
6. Export to Excel for further analysis
```

**Output**: Publication-quality dataset with < 1 ns uncertainty

### Workflow 3: Long-Term Stability (24 hours)

```
1. Configure for 1440 captures (1 per minute)
2. Disable screenshots (saves space)
3. Enable waveform CSV
4. Run overnight
5. Analyze delay drift over time
```

**Output**: Time-series showing delay stability

---

## 🎯 Use Cases

### ✅ Perfect For:
- Laser-APD delay characterization
- Timing jitter measurement
- Signal correlation analysis
- System stability testing
- Optical delay verification
- Research & development

### ❌ Not Suitable For:
- Single-shot captures (use scope directly)
- Real-time feedback control (this is offline analysis)
- Non-repetitive signals
- Signals without correlation

---

## 📞 Support

**Having issues?**

1. Check [ENHANCED_VERSION_GUIDE.md](ENHANCED_VERSION_GUIDE.md) - Troubleshooting section
2. Verify dependencies: `pip list`
3. Test connection: Connection tab → "Test Connection"
4. Check oscilloscope: Verify signals visible manually

**Common fixes:**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements_enhanced.txt

# Verify installation
python -c "import scipy; print('scipy OK')"

# Test basic import
python -c "from continuous_trigger_capture_enhanced import *"
```

---

## 📄 License & Credits

**Developed by**: Anirudh Iyengar
**Organization**: Digantara Research and Technologies Pvt. Ltd.
**Purpose**: Internal research & development tool

**Third-party libraries**:
- NumPy, SciPy, Pandas: BSD License
- Matplotlib: PSF License
- Gradio: Apache 2.0
- PyVISA: MIT License

---

## 🎓 Learn More

### Scientific Background
- **Cross-correlation**: Measures similarity between two signals
- **Jitter**: Timing variations in periodic signals
- **RMS vs Pk-Pk**: RMS = typical, Pk-Pk = worst-case
- **Confidence intervals**: Uncertainty quantification

### Further Reading
- [ENHANCED_VERSION_GUIDE.md](ENHANCED_VERSION_GUIDE.md) - Complete 60-page manual
- Bendat & Piersol - "Random Data Analysis"
- Taylor - "Error Analysis"

---

## ✅ Quick Verification

After installation, verify everything works:

```bash
# 1. Test imports
python -c "import numpy, pandas, matplotlib, scipy, gradio; print('OK')"

# 2. Launch application
python continuous_trigger_capture_enhanced.py

# 3. Web interface should open automatically
# 4. Go to Connection tab
# 5. Try connecting to your scope
```

If you see the web interface, you're ready! 🎉

---

## 🚀 What's Next?

1. ✅ Install dependencies ← **Start here**
2. 📖 Read [INSTALL_INSTRUCTIONS.md](INSTALL_INSTRUCTIONS.md)
3. 🎮 Run Quick Start example
4. 📊 Capture your first dataset
5. 📈 Analyze results
6. 📚 Explore advanced features in [ENHANCED_VERSION_GUIDE.md](ENHANCED_VERSION_GUIDE.md)

---

## 🎉 Summary

**This enhanced system gives you:**

✨ **Automated** delay measurement (no manual analysis)
✨ **Real-time** statistics and visualization
✨ **Professional** reports and exports
✨ **Easy** to use with Quick Start interface
✨ **Accurate** cross-correlation algorithm
✨ **Complete** documentation and guides

**Perfect for precision laser-APD timing measurements! 🔬**

---

**Ready to measure? Launch the app and start capturing! 🚀**

```bash
python continuous_trigger_capture_enhanced.py
```
