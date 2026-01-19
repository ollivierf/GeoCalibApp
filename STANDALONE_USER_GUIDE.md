# GeoCalibApp - Standalone Application User Guide

## Overview
This document provides instructions for using the standalone Windows application **GeoCalibApp**, designed for geometric calibration of microphone arrays using Time-of-Arrival (TOA) measurements. This version does not require Python to be installed.

## 1. Installation & Setup
The application is provided as a standalone **folder**. 

1.  **Unpack**: Ensure you have the entire `GeoCalibApp` folder. 
2.  **Do Not Separate Files**: The executable `GeoCalibApp.exe` relies on the other files in the directory (such as `_internal`). **Do not move the .exe file out of this folder.** You can move the entire folder to any location (e.g., Desktop).
3.  **Antivirus Warning**: On first launch, Windows Defender or other antivirus software might flag the application because it is not digitally signed. You may need to click "More Info" -> "Run Anyway" or create an exception.

## 2. Launching the Application
*   Navigate to the `GeoCalibApp` folder.
*   Double-click **`GeoCalibApp.exe`**.
*   The application window should appear immediately.

## 3. Workflow

The application supports a vertical scrollable workflow. You can scroll down to access subsequent steps.

### Step 1: Time-of-Arrival (TOA) Extraction

> **WARNING: Data Format Requirement**  
> The input data files must be in **Mµ format** (`.dat` or `.h5`). They are expected to contain:
> *   The **Counter**.
> *   The **analog signal transmitted to the source** (ideally a swept sine) on the **first analog channel**.

1.  **Select Files**: Click **"Select Data Files"** to choose your measurement files (`.h5` or `.dat`).
    *   *Note*: `.h5` files are the preferred standard.
2.  **Process**: Click **"Process Selected Files"**.
    *   A dialog will appear for each file showing the Cross-Correlation (GCC-PHAT) heatmap.
    *   **Validation**: Use the sliders to define the valid distance range (Min/Max).
    *   **Validate**: Click "Validate" to accept the TOA extraction for the file.
3.  **Review**: The TOA matrices will be loaded. Use the **"Show TOA Maps"** button to visualize the extraction quality.
4.  **Selection**: Check the boxes next to the files you want to include in the geometry solving.

### Step 2: Geometry Solver
1.  **Configure Parameters**:
    *   **Max Iterations**: Default is `20000`. Increase if convergence is not reached.
    *   **Epsilon (10^-x)**: Convergence threshold. Default input `9` means $10^{-9}$. 
    *   **Temperature**: Adjust if measurement conditions differed from `26.0°C`.
2.  **Run Solver**: Click **"Run RMDU Optimization"**.
    *   The 3D plot will update in real-time.
    *   Watch the **Error Evolution** graph for convergence (slope flattening).
3.  **Convergence**: The process stops automatically when the error change drops below epsilon or max iterations are reached.

### Step 3: Alignment & Export
1.  **Reference Microphones**: If you have known anchor points, enter their indices and coordinates here.
2.  **Align**: Click **"Align Geometry"** to transform the solved "cloud" of points to your coordinate system using Procrustes analysis.
3.  **Export Results**:
    *   **Save Geometry**: Exports the XYZ coordinates to `.npy` or `.csv`.
    *   **Save Report**: Generates a summary of the calibration error.

## 4. Troubleshooting

*   **App Closes Immediately**: If the application crashes on startup, ensure you are running it from inside its folder and that you have write permissions to that folder.
*   **"Permission Denied"**: Do not run the application from inside a ZIP file. Extract it first.
*   **Performance**: Real-time 3D plotting can be intensive. If the interface lags during solving, this is normal behavior as it updates the visualization.

## 5. File Formats
*   **HDF5 (.h5)**: Standard input containing raw audio data.
*   **DAT/LOG**: Legacy binary formats supported via the internal parser.

---
**Support**: For technical issues, verify that all files in the `_internal` folder are present and unmodified.
