# GeoCalib - Complete User Manual

**Comprehensive reference for all features and workflows**

---

## Table of Contents
1. [Quick Start (5 minutes)](#quick-start)
2. [Step 1: TOA Processing](#step-1-toa-processing)
3. [Step 2: Geometry Solving](#step-2-geometry-solving)
4. [Step 3: Alignment & Export](#step-3-alignment-export)
5. [GCC Validation Details](#gcc-validation)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Option 1: Standalone Application (Recommended)
**No installation required.**

1.  **Download/Copy** the provided `GeoCalibApp` folder.
2.  **Open** the folder and double-click `GeoCalibApp.exe`.
3.  **Start** using the application.

*Note: Do not move the `.exe` file out of the folder. For more details, see [STANDALONE_USER_GUIDE.md](STANDALONE_USER_GUIDE.md).*

### Option 2: Running from Source (Python)

**5-Minute Setup**

```bash
# 1. Navigate to project folder
cd CalibGeo/PyqtApp

# 2. Create + activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1          # Windows
# source venv/bin/activate            # Linux/macOS

# 3. Install packages
pip install -r requirements.txt

# 4. Run app
python GeoCalibApp.py
```

**First Use Checklist**:
- [ ] Identify your audio files (HDF5 .h5 or DAT+LOG)
- [ ] Know room temperature in °C
- [ ] Have reference microphone positions ready
- [ ] Open GeoCalibApp and follow 3 tabs

---

## Step 1: TOA Processing

### Overview
Extracts Time-of-Arrival (TOA) values from audio files using cross-correlation.

> **WARNING: Data Format Requirement**  
> The input data files must be in **Mµ format** (`.dat` or `.h5`). They are expected to contain:
> *   The **Counter**.
> *   The **analog signal transmitted to the source** (ideally a swept sine) on the **first analog channel**.

**Input**: Audio files (HDF5 or DAT format)  
**Process**: GCC-PHAT cross-correlation → Peak detection → Interactive validation  
**Output**: TOA values for each microphone

### Detailed Workflow

#### 1.1 Select Audio Files

```
┌─────────────────────────────────┐
│ STEP 1: TOA PROCESSING          │
├─────────────────────────────────┤
│ Files: [Browse Files]           │
│ ├─ audio_001.h5                 │
│ ├─ audio_002.h5                 │
│ └─ recording.dat/log            │
│                                 │
│ Parameters:                     │
│ Temperature: [26]°C             │
│ Max Distance: [10.0]m           │
│                                 │
│ [Process TOA Files]             │
└─────────────────────────────────┘
```

**Actions**:
1. Click **"Browse Files"**
2. Select all audio files (multi-select OK)
3. Files appear in list below
4. Verify file count

**Supported Formats**:
- `.h5`, `.hdf5` - HDF5 format (recommended)
- `.dat` + `.log` pair - DAT format (must have both files)

#### 1.2 Configure Parameters

**Temperature (Required)**
- **What**: Room air temperature in Celsius
- **Why**: Affects sound speed calculation
- **Formula**: C = √(1.4 × 287 × (T + 273)) m/s
- **Examples**:
  - 20°C: 343 m/s
  - 26°C: 346 m/s
  - 30°C: 349 m/s
- **Enter**: Best guess OK (±1°C acceptable)

**Max Distance (Optional)**
- **What**: Maximum search range for cross-correlation
- **Default**: 10.0 m
- **Range**: 0.1 - 100.0 m
- **Use**: Speeds up computation if you know max array size

#### 1.3 Process TOA Files

```
Click [Process TOA Files]
↓
For each file:
  → Reads audio data
  → Computes GCC-PHAT cross-correlation
  → Extracts envelope (Hilbert transform)
  → Detects peaks
  → Launches GCC Validation Dialog ← INTERACTIVE VALIDATION
  ↓
You review dialog:
  → See greyscale heatmap
  → Adjust distance sliders
  → Click "Use This Range"
  ↓
TOA values computed within selected distance range
Processing continues to next file
```

**Status Bar** shows:
- Current file being processed
- Overall progress
- Estimated time remaining

### 1.4 GCC Validation Dialog (Interactive)

#### What You See

```
GCC VALIDATION - file_001.h5
┌────────────────────────────────────────────┐
│ GCCE Heatmap (0-10 meters)                 │
│                                            │
│ 256 ├───●●────────────────────────┐      │
│     │  ●●●●●                      │      │
│ 128 │  ●●●●●●●                    │  ← Red dots: Detected TOA peaks
│     │    ●●●●●●●●●                │      │
│   0 └────────────────────────────┘      │
│     0                           0.03s      │
│     ├─────────────────┼──┤               │
│     Min Distance       Max Distance       │
│     (green line)       (red line)         │
│                                            │
│ Distance Range: 1.5m - 8.2m               │
│ Peaks in range: 234/256                   │
│ Mean TOA: 0.0249s                         │
│ Std Dev: 0.0008s                          │
│                                            │
│ [Min: 0────5──10m]  [Max: 0────5──10m]   │
│                                            │
│ [ Cancel ]  [ Use This Range ]            │
└────────────────────────────────────────────┘
```

#### Components Explained

**Greyscale Heatmap**:
- **Horizontal axis**: Microphone index (0-256)
- **Vertical axis**: Time delay in seconds (0 to ~0.03s at 26°C)
- **Brightness**: Correlation strength
  - Dark: No energy
  - Bright: Strong correlation
  - Peak = direct sound arrival

**Red Dots**: Detected TOA peaks
- One dot per microphone
- Position = estimated arrival time
- Cluster = good signal quality
- Scattered = noisy or weak signal

**Range Lines**:
- **Green line**: Minimum distance threshold
- **Red line**: Maximum distance threshold
- Sliders control these independently

**Statistics Panel**:
| Stat | Meaning |
|------|---------|
| Distance Range | Min-max selected (meters) |
| Peaks in range | Count of peaks within bounds |
| Mean TOA | Average time-of-arrival (s) |
| Std Dev | Spread of TOA values |

#### How to Use Sliders

**Goal**: Isolate the valid TOA peaks

**Scenario 1: Clean Signal**
```
Red dots cluster tightly around one distance
→ Bright band in heatmap at that distance
→ Sliders naturally frame the cluster
→ Accept range
```

**Scenario 2: Weak Signal with Noise**
```
Red dots scattered, some outliers
→ Heatmap diffuse with bright central region
→ Use sliders to exclude outliers
→ Focus on region with highest density
→ Tighten range around main cluster
```

**Scenario 3: Multi-path (Echoes)**
```
Multiple bright bands in heatmap
→ Multiple red dot clusters
→ Select the STRONGEST cluster (brightest band)
→ Set sliders to include only direct path
→ Exclude weaker echoes
```

**Step-by-Step Adjustment**:
1. Start: Both sliders at extremes (0-10m)
2. Look: Where are most red dots concentrated?
3. Adjust **Min slider**: Move right to remove low outliers
4. Adjust **Max slider**: Move left to remove high outliers
5. Goal: Tighten range to high-density region
6. Review: Statistics panel shows peak count in range
7. Accept: Click "Use This Range"

**Pro Tips**:
- If all peaks at large distance → room reflections, check signal
- If all peaks at small distance → near-field source, normal
- If scattered peaks → consider acoustic noise, re-record
- If zero peaks in range → move sliders, something wrong

#### Accepting the Range

Click **"Use This Range"**:
- TOA values are clamped to selected distance bounds
- Out-of-range values set to nearest bound
- Processing continues to next file
- Dialog closes automatically

Click **"Cancel"**:
- Skip this file entirely
- No TOA values computed
- Move to next file
- Can re-process later

### 1.5 DTOA Validation Dialog

After all files processed:

```
DTOA VALIDATION
┌──────────────────────────────┐
│ Processed 5 files            │
│                              │
│ Files with good TOA:         │
│ ☑ audio_001.h5 (234/256)    │
│ ☑ audio_002.h5 (251/256)    │
│ ☑ audio_003.h5 (248/256)    │
│ ☐ audio_004.h5 (156/256)    │
│ ☑ audio_005.h5 (255/256)    │
│                              │
│ DTOA Heatmaps (5 pairs):     │
│ [01-02] [01-03] [01-04]      │
│ [02-03] [02-04]              │
│                              │
│ [ Cancel ]  [ Use Selected ] │
└──────────────────────────────┘
```

**What to Do**:
1. Review DTOA heatmaps for each file pair
2. Look for clear diagonal patterns (good DTOA)
3. Deselect files with noisy patterns
4. Click "Use Selected" to keep good files
5. Continue to Step 2

---

## Step 2: Geometry Solving

### Overview
Uses Regularized Damped Update (RMDU) solver to compute 3D microphone positions from differential TOA values.

**Input**: Validated DTOA values from Step 1  
**Process**: Iterative optimization with real-time visualization  
**Output**: 3D positions (X, Y, Z in meters)

### 2.1 Solver Parameters

```
┌─────────────────────────────────┐
│ STEP 2: GEOMETRY SOLVING        │
├─────────────────────────────────┤
│ Solver Parameters:              │
│                                 │
│ Lambda: [0.050 ─────○──────]    │ ← Damping factor
│ Max Iterations: [10000]          │ ← Iteration limit
│ Epsilon Limit: [1e-10]           │ ← Convergence criterion
│                                 │
│ [Run Solver]                    │
│                                 │
│ Status: Ready                   │
│ Progress: ──────────────        │
│ Iterations: 0/10000             │
└─────────────────────────────────┘
```

**Parameter Guide**:

| Parameter | Min | Default | Max | Meaning |
|-----------|-----|---------|-----|---------|
| Lambda | 0.001 | 0.050 | 0.100 | Damping/stabilization (smaller = less stable, faster) |
| Max Iterations | 100 | 10000 | 100000 | Upper limit on optimization steps |
| Epsilon | 1e-12 | 1e-10 | 1e-6 | Convergence tolerance (smaller = tighter) |

**Quick Start**: Use defaults. Adjust only if:
- **Converges too slowly**: Reduce Lambda to 0.020, increase Max Iterations
- **Oscillates/unstable**: Increase Lambda to 0.100
- **Insufficient accuracy**: Decrease Epsilon to 1e-12

### 2.2 Run Solver

```
Click [Run Solver]
↓
Solver starts
Real-time display:
  - Progress bar: Current iteration / Max
  - 3D View: Array geometry updates in real-time
  - Iteration Plot: Error vs iteration (decreasing line)
  - Status: Current error value
↓
Converges when:
  - Error change < Epsilon (good stopping)
  - Max iterations reached (may need adjustment)
↓
Solver stops automatically
```

**What's Happening**:
1. RMDU optimizer minimizes difference between:
   - Measured DTOA values
   - Computed DTOA from current geometry guess
2. Each iteration:
   - Computes residual error
   - Updates microphone positions
   - Visualizes result
3. Converges when error stabilizes

**Monitoring**:
- **Iteration Plot**: Should show smooth downward curve
- **3D View**: Array should show recognizable shape
- **Status**: Error value decreasing → healthy

### 2.3 Review Results

```
Convergence Summary:
- Final Error: 0.000234 meters
- Iterations: 8,432 / 10,000
- Convergence: Epsilon criterion met ✓
- Computation Time: 23 seconds

3D View Shows:
  - Microphone array as point cloud
  - Roughly spherical or linear arrangement
  - No wild outliers (>2m from median)
  - Symmetric if reference was symmetric
```

**Red Flags** (solver problems):
- Error doesn't decrease → Bad DTOA data, try different files
- Mics far apart with huge error → Reference offset wrong
- Oscillating error → Increase Lambda parameter

**Good Signs**:
- Smooth exponential error decay
- Final error < 0.5mm
- Recognizable array structure
- Convergence < 50% iterations

### 2.4 Visualization Controls

**3D View**:
- **Left drag**: Rotate view
- **Right drag**: Pan view
- **Scroll**: Zoom in/out
- **Space bar**: Reset view

**Iteration Plot**:
- **Play button**: Animate solver progress
- **Slider**: Jump to iteration N
- **Save**: Export plot as image

---

## Step 3: Alignment & Export

### Overview
Transforms computed array to reference coordinate frame using Procrustes alignment.

**Input**: 3D positions from Step 2 + reference positions  
**Process**: Rotation + translation to match reference  
**Output**: Aligned positions, JSON config, CSV export

### 3.1 Enter Reference Positions

```
┌─────────────────────────────────┐
│ STEP 3: ALIGNMENT & EXPORT      │
├─────────────────────────────────┤
│                                 │
│ Reference Position Input:       │
│ [Enter Ref Positions]           │
│                                 │
│ Dialog:                         │
│ Microphone 1: X= [0.50] Y=[0.50] Z=[1.50]
│ Microphone 2: X= [0.50] Y=[-0.50] Z=[1.50]
│ Microphone 3: X= [-0.50] Y=[0.50] Z=[1.50]
│ Microphone 4: X= [-0.50] Y=[-0.50] Z=[1.50]
│                                 │
│ [ OK ]  [ Cancel ]              │
└─────────────────────────────────┘
```

**What**: Precise positions of 4-6 reference microphones (usually outer mics)

**Why**: Aligns computed array to your coordinate system

**How to Measure**:
1. Choose 4 reference mics (usually corners of array)
2. Measure positions relative to origin
3. Use meter stick or laser measure
4. Record X, Y, Z in meters

**Template Format** (load from `reference_microphones.json`):
```json
{
  "reference_points": [
    {"index": 1, "x": 0.5, "y": 0.5, "z": 1.5},
    {"index": 64, "x": 0.5, "y": -0.5, "z": 1.5},
    {"index": 192, "x": -0.5, "y": 0.5, "z": 1.5},
    {"index": 256, "x": -0.5, "y": -0.5, "z": 1.5}
  ]
}
```

**Quick Entry**:
1. Load template: `[Load Template]`
2. Edit values: Click each field, change X/Y/Z
3. Accept: Click `[OK]`

### 3.2 Select Microphone Indices

```
Mic Indices Dialog:

Which array mics match the reference positions?

Ref Position 1 (0.5, 0.5, 1.5) → Array Mic [  64  ]
Ref Position 2 (0.5, -0.5, 1.5) → Array Mic [ 192  ]
Ref Position 3 (-0.5, 0.5, 1.5) → Array Mic [  48  ]
Ref Position 4 (-0.5, -0.5, 1.5) → Array Mic [ 208  ]

[ OK ]  [ Cancel ]
```

**What**: Match 4 reference positions to actual microphone indices

**Why**: Pairs computed positions with measured references

**How**:
1. For each reference position, find matching mic in array
2. Enter microphone index number (1-256)
3. Check: First ref should match first mic index, etc.
4. Order matters!

**Example**:
```
Ref Pos 1 → Mic 64 (front-left)
Ref Pos 2 → Mic 192 (front-right)
Ref Pos 3 → Mic 48 (back-left)
Ref Pos 4 → Mic 208 (back-right)
```

### 3.3 Perform Alignment

```
Click [Perform Alignment]
↓
Procrustes Algorithm:
1. Compares reference positions → microphone positions
2. Computes rotation matrix (best-fit orientation)
3. Computes translation vector (position offset)
4. Applies both to entire array
↓
Results:
- All 256 positions transformed
- Array now in your reference frame
- Original structure preserved
- Ready for export
```

**What's Computed**:
- **Rotation**: 3×3 matrix to align orientation
- **Translation**: 3-vector offset to align position
- **Error**: RMS difference after alignment

**Results Display**:
```
Alignment Results:
- Rotation matrix: [3×3]
- Translation: [x, y, z]
- Alignment error: 0.012 m
- Status: ✓ Successful
```

### 3.4 Export Results

```
Buttons:
[Save XYZ]   → Saves aligned positions as .npy file
[Save Config] → Saves JSON with parameters
[Save CSV]   → Saves positions in spreadsheet format
[Done]       → Closes application
```

**Save XYZ**:
- NumPy binary format (.npy)
- Load in Python: `xyz = np.load('positions.npy')`
- Shape: (256, 3) = 256 microphones × 3 coordinates
- Units: Meters

**Save Config**:
- JSON file with all processing parameters
- Includes: Temperature, file list, solver params, alignment matrix
- Use for: Reproducibility, documentation

**Save CSV** (Optional):
- Human-readable spreadsheet format
- Columns: Mic#, X, Y, Z
- Units: Meters

---

## GCC Validation

### Understanding GCC-PHAT

**What is GCC?**
- Generalized Cross-Correlation (GCC)
- Compares audio channels pairwise
- Finds time delay between channels
- Proportional to microphone distance from sound source

**What is PHAT?**
- Phase Alignment Transform
- Prefilter for GCC
- Emphasizes strong peaks in time-domain
- Reduces effects of noise and reverberation

**Why GCC-PHAT?**
- Fast computation
- Robust to noise
- Works in reverberant environments
- Industry standard for microphone arrays

### GCC Computation Pipeline

```
Audio Input (multichannel WAV/HDF5)
    ↓
Extract reference channel (mic 0)
    ↓
For each channel i (1-256):
    ├─ Compute cross-correlation: CC(τ) = ∫ ref(t)·mic_i(t+τ) dt
    ├─ Apply PHAT weighting: CC_PHAT(τ) = CC(τ) / |CC(τ)|
    └─ Find peak τ_i = argmax(CC_PHAT)
    ↓
Extract envelope (Hilbert transform):
    ├─ GCCE = |Hilbert(CC_PHAT)|
    ├─ Normalize 0-1 for visualization
    └─ Create greyscale heatmap
    ↓
Detect peaks:
    ├─ Find local maxima in each row
    ├─ Skip first/last samples (edge artifacts)
    └─ Place red dots on heatmap
    ↓
Launch GCC Validation Dialog
    ↓
User selects distance range (min-max sliders)
    ↓
Compute TOA from distance:
    ├─ TOA = distance / sound_speed
    ├─ Sound speed from temperature
    └─ Values outside range → clamp to bounds
    ↓
TOA array ready for Geometry Solving
```

### Reading the Heatmap

#### Signal Quality Indicators

**Excellent Signal**:
```
Red dots: Tight cluster
Heatmap: Bright narrow band
Pattern: All mics show same delay
→ Direct sound, good SNR
```

**Good Signal**:
```
Red dots: Mostly clustered, few outliers
Heatmap: Bright region with faint edges
Pattern: 90%+ mics agree on delay
→ Some noise, acceptable
```

**Weak Signal**:
```
Red dots: Scattered
Heatmap: Diffuse, low brightness
Pattern: Mics scattered over range
→ Low SNR, try narrower window
```

**Multi-path (Room Reflections)**:
```
Red dots: Multiple clusters
Heatmap: Multiple bright bands
Pattern: Some mics at ~1m, some at ~2m
→ Select strongest cluster (direct path)
```

### Distance Calibration

**Key Formula**:
```
TOA = Distance / Sound_Speed

Sound Speed (meters/second):
C = √(1.4 × 287 × (T + 273))

Where T = Temperature in °C
```

**Example Calculations** (at 26°C, C = 346 m/s):
```
Distance | Time Delay
0.5 m    | 1.45 ms
1.0 m    | 2.89 ms
2.0 m    | 5.78 ms
5.0 m    | 14.45 ms
10.0 m   | 28.90 ms
```

**Verifying Distance Sliders**:
1. Know distance to sound source (measure tape)
2. Expected TOA = distance / sound_speed
3. Check if red dots at that location
4. Adjust sliders to frame the distance

---

## Troubleshooting

### TOA Processing Issues

**Problem: "Error reading HDF5 file"**
```
Cause: File corrupted or wrong format
Solution:
1. Verify file is valid HDF5 (try opening in HDFView)
2. Check file isn't corrupted (copy fresh from source)
3. Use different file to test
```

**Problem: "No valid TOA peaks detected"**
```
Cause: 
- Signal too weak
- File format unrecognized
- Wrong channel layout

Solution:
1. Check file format (HDF5 or DAT+LOG pair)
2. Verify audio levels (open in Audacity)
3. Try with known-good file
4. Increase max distance slider in validation dialog
```

**Problem: GCC Validation dialog doesn't appear**
```
Cause: 
- Window rendered off-screen
- Display scaling issue
- Dialog lost focus

Solution:
1. Restart application
2. Check display settings (100% scaling)
3. Try different screen resolution
4. Run: Python GeoCalibApp.py (check console for errors)
```

**Problem: All red dots at distance < 0.5m**
```
Cause: Microphones very close to source OR false peak detection
Solution:
1. Check source distance (should be >1m for array)
2. Narrow min slider to exclude false peaks
3. Look at heatmap brightness (false peaks are faint)
```

### Geometry Solving Issues

**Problem: "Solver not converging" (error stays high)**
```
Cause:
- Poor DTOA data quality
- Wrong reference positions
- Outlier mics included

Solution:
1. Check DTOA validation heatmaps for patterns
2. Remove files with scattered peaks
3. Verify reference mic indices are correct
4. Increase max iterations to 20000
5. Decrease lambda to 0.020 for faster convergence
```

**Problem: "Microphones scattered far apart"**
```
Cause: 
- Bad DTOA data
- Reference positions wrong
- Solver diverging

Solution:
1. Return to Step 1, review TOA validation
2. Check reference_microphones.json for typos
3. Measure reference positions again
4. Use subset of best-quality files
```

**Problem: Computed geometry looks like straight line (not array)**
```
Cause: All DTOA values same (source very far away) OR all files same delay
Solution:
1. Move sound source closer (5-10m typical)
2. Vary source position (multiple files at different positions)
3. Check temperature setting (wrong C value?)
```

### Alignment Issues

**Problem: "Alignment error too large" (>0.5m)**
```
Cause:
- Reference positions measured wrong
- Mic indices don't match reference positions
- Geometry doesn't match reference scale

Solution:
1. Re-measure reference positions with meter stick
2. Double-check mic indices (use physical labeling)
3. Verify scale: Computed array should be ~1m, not 100m
```

**Problem: Aligned array oriented wrong**
```
Cause: Reference positions have different orientation than measurement
Solution:
1. Define reference frame clearly (which axis is X? Y?)
2. Measure all positions in same frame
3. Rotation is automatically computed, this is OK
4. Check final positions make sense
```

### General Troubleshooting

**Problem: Application crashes during processing**
```
Cause: Out of memory or unhandled exception
Solution:
1. Check system RAM available
2. Close other applications
3. Restart Python from command line
4. Check console for error messages
5. Report error with full stack trace
```

**Problem: "rcbox not available"**
```
Cause: RMDU solver library not installed
Solution:
pip install rcbox
Or manually copy rcbox/ folder to application directory
```

**Problem: Processing very slow**
```
Cause: Large files or high CPU load
Solution:
1. Close other applications
2. Reduce max distance in Step 1 (speeds GCC)
3. Use fewer files (process 5 at a time)
4. Lower max iterations in Step 2 (speeds solver)
```

---

## Frequently Asked Questions

**Q: How many files should I process?**
A: At least 3-5 for robust geometry. More files = better accuracy.

**Q: What if some files have poor TOA?**
A: Skip them in DTOA validation dialog (uncheck). Use only best-quality files.

**Q: Can I restart midway through?**
A: Yes. Each step saves its output. You can reload in next session.

**Q: How accurate are final positions?**
A: Typically 1-5mm error in controlled lab. Depends on signal quality and reference accuracy.

**Q: Why is temperature important?**
A: Sound speed changes ~0.6 m/s per °C. Wrong T → wrong distances → inaccurate positions.

**Q: What if I don't know exact room temperature?**
A: Estimate (±1-2°C). Calibration process accounts for some uncertainty.

**Q: Can I use fewer reference points?**
A: Minimum 4 for Procrustes alignment. More = better constraint.

**Q: What file formats are supported?**
A: HDF5 (.h5, .hdf5) or DAT+LOG pairs. Other formats need custom importer.

---

## Quick Reference Cards

### Slider Interpretation
```
Min Distance Slider:
Left (0m)   → Include all distances
Right (10m) → Exclude distant sources

Max Distance Slider:
Left (0m)   → Include nothing
Right (10m) → Include all distances

Goal: Tighten range to high-density peak region
```

### Temperature→Sound Speed
```
15°C = 340 m/s
20°C = 343 m/s
25°C = 345 m/s
26°C = 346 m/s ← Typical room temp
30°C = 349 m/s
35°C = 352 m/s
```

### Solver Parameter Quick Tuning
```
Converges too slow:
  ↓ Lambda (0.020)
  ↑ Max Iterations (20000)

Oscillates/unstable:
  ↑ Lambda (0.100)

Not accurate enough:
  ↓ Epsilon (1e-12)
```

---

**Version 1.0 - Complete Documentation**  
**For questions, see DEPLOYMENT_GUIDE.md or contact your system administrator**
