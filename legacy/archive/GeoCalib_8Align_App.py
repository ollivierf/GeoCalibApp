#!/usr/bin/env python3
"""
GeoCalib - Microphone Array Geometric Calibration Application

A comprehensive PyQt5 tool for computing 3D microphone positions from Time-of-Arrival (TOA)
measurements using advanced optimization algorithms (RMDU solver with Procrustes alignment).

FEATURES:
  • Multi-format input: HDF5 (.h5) and binary DAT + LOG files
  • TOA extraction: GCC-PHAT cross-correlation algorithm
  • Geometry solver: RMDU iterative optimization with convergence display
  • Alignment: Procrustes transformation for geometry alignment
  • Visualization: Real-time 3D rendering, TOA heatmaps, convergence plots
  • Batch processing: Multiple files with selective TOA map inclusion
  • Export: NumPy arrays, JSON configurations, results summary

WORKFLOW:
  1. Step 1 (TOA): Select files → Process → Review maps → Select valid data
  2. Step 2 (Solver): Configure parameters → Run optimization → Monitor convergence
  3. Step 3 (Alignment): Align to reference → Transform geometry → Export results

USAGE:
  python GeoCalibApp.py

DEPENDENCIES:
  • PyQt5 5.15+: GUI framework
  • numpy, scipy: Numerical computing
  • h5py: HDF5 file support
  • pyqtgraph 0.14+: 3D visualization
  • rcbox: Local RMDU solver (pre-compiled, included)
  • matplotlib: Plotting (optional)

VERSION: 1.0
LAST UPDATED: 2025-12-18
"""

import sys, os, json, hashlib, pickle, threading
import numpy as np
from datetime import datetime
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QFileDialog, QTabWidget, 
    QTableWidget, QTableWidgetItem, QComboBox, QProgressBar, QGroupBox, QFormLayout, 
    QCheckBox, QMessageBox, QListWidget, QListWidgetItem, QDialog, QSlider, QStatusBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QColor, QFont
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import h5py
import scipy.signal as sig
import warnings
import matplotlib as plt
try:
    from rcbox.rmds import RMDU, _Lmake, compute_Lpinv, _Slambda
    from scipy.spatial.distance import squareform, pdist
    import numpy.linalg as linalg
    RCBOX_AVAILABLE = True
except ImportError:
    RCBOX_AVAILABLE = False

try:
    from DATParser import DATParser
    DATPARSER_AVAILABLE = True
except ImportError:
    DATPARSER_AVAILABLE = False

from GeoCalibUtils import procrustes, evalDTOA
warnings.filterwarnings("ignore")

class GCCValidationDialog(QDialog):
    """\n    Interactive dialog for validating and selecting Time-of-Arrival (TOA) range from GCC-PHAT.
    
    Displays greyscale GCCE (envelope) heatmap with detected peaks marked as red dots.
    Allows user to interactively set min/max distance constraints and recompute TOA values.
    Seamlessly processes multiple files with user validation at each step.
    
    Features:
        • Greyscale GCCE heatmap (0-10 meters)
        • Red dots marking detected peaks (TOA maxima)
        • Min/Max distance sliders with real-time update
        • Visual range guides (vertical lines at min/max)
        • Live TOA recomputation on slider changes
    """
    
    # Signal to indicate file should be discarded
    file_discarded = pyqtSignal(str)  # Emits filename
    
    def __init__(self, filename, gcce, tt, toas, imax, temp_celsius=26, parent=None):
        super().__init__(parent)
        self.filename = filename
        self.gcce = gcce  # (NbMics, NbSamples)
        self.tt = tt  # Time array
        self.toas = toas.copy()  # Initial TOA values
        self.imax = imax.copy()  # Initial peak indices
        self.Tc = temp_celsius
        self.C = np.sqrt(1.4 * 287 * (self.Tc + 273))
        
        # Extract valid time range (max 10 meters)
        self.tmax = 10.0 / self.C  # Maximum time for 10 meters
        valid_idx = self.tt < self.tmax
        self.gcce_limited = gcce[:, valid_idx]
        self.tt_limited = tt[valid_idx]
        
        self.min_length = 0.0
        self.max_length = 10.0
        self.confirmed = False
        
        self.setWindowTitle(f"GCC Validation - {filename}")
        self.setGeometry(100, 100, 1400, 600)
        self.init_ui()
    
    def init_ui(self):
        """Initialize validation dialog UI"""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel(f"GCC-PHAT Validation: {self.filename}")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Main content: heatmap + controls
        content_layout = QHBoxLayout()
        
        # Left: GCCE heatmap
        heatmap_group = QGroupBox("GCCE Magnitude (0-10m)")
        heatmap_layout = QVBoxLayout()
        
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('bottom', 'Microphone Index')
        self.plot_widget.setLabel('left', 'Time (ms)')
        self.plot_widget.setTitle('GCC-PHAT Energy Envelope (All Microphones)')
        
        # Store reference to left axis for later use
        self.left_axis = self.plot_widget.getAxis('left')
        
        # Get the plot layout to add secondary Y-axis
        plot_layout = self.plot_widget.plotItem.layout
        
        # Add secondary Y-axis for distance
        self.right_axis = pg.AxisItem(orientation='right')
        self.right_axis.setLabel('Distance (m)', color='white')
        plot_layout.addItem(self.right_axis, 2, 3)
        
        # Link the right axis to the plot's view box
        self.right_axis.linkToView(self.plot_widget.getViewBox())
        
        # Flip so time increases from bottom to top and index left to right
        gcce_rotated = self.gcce_limited
        #gcce_rotated = np.flipud(gcce_rotated)
        
        # Display GCCE as image with proper scaling
        self.img_item = pg.ImageItem(image=gcce_rotated)
        
        # Set image position and scale to match data coordinates
        # Image shape is (NbSamples, NbMics), mapped to:
        # X-axis: 0 to NbMics (microphone indices)
        # Y-axis: 0 to max(tt_limited) (time)
        nb_mics = self.gcce_limited.shape[0]
        nb_samples = len(self.tt_limited)
        
        # Use setRect to set position and scale properly
        # QRectF(x, y, width, height)
        # Rect coordinates: time runs from bottom (0) to top (max_time)
        from PyQt5.QtCore import QRectF
        rect = QRectF(0, 0, nb_mics, self.tt_limited.max())
        self.img_item.setRect(rect)
        
        self.plot_widget.addItem(self.img_item)
        
        # Set colormap to greyscale
        colormap = pg.ColorMap(pos=np.array([0.0, 1.0]), 
                               color=np.array([[0, 0, 0, 255], [255, 255, 255, 255]], dtype=np.ubyte))
        self.img_item.setColorMap(colormap)
        self.img_item.setLevels([gcce_rotated.min(), gcce_rotated.max()])
        
        # Plot peak markers (red dots) - adjusted for rotated coordinates
        self.peak_scatter = pg.ScatterPlotItem()
        self.update_peak_markers()
        self.plot_widget.addItem(self.peak_scatter)
        
        # Add range lines - now horizontal lines at time positions
        self.line_min = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('c', width=1, style=Qt.DashLine))
        self.line_max = pg.InfiniteLine(pos=self.tt_limited.max(), angle=0, pen=pg.mkPen('c', width=1, style=Qt.DashLine))
        self.plot_widget.addItem(self.line_min)
        self.plot_widget.addItem(self.line_max)
        
        # Set axes ranges
        self.plot_widget.setXRange(0, self.gcce_limited.shape[0])  # NbMics on x-axis
        self.plot_widget.setYRange(0, self.tt_limited.max())  # Time on y-axis (0 at bottom, max at top)
        
        # Update distance axis labels based on time values
        self.update_distance_axis()
        
        heatmap_layout.addWidget(self.plot_widget)
        heatmap_group.setLayout(heatmap_layout)
        content_layout.addWidget(heatmap_group, 3)
        
        # Right: Controls
        control_group = QGroupBox("Distance Controls")
        control_layout = QVBoxLayout()
        
        # Min distance
        min_layout = QVBoxLayout()
        min_layout.addWidget(QLabel("Minimum Distance (m):"))
        self.min_slider = QDoubleSpinBox()
        self.min_slider.setMinimum(0.0)
        self.min_slider.setMaximum(10.0)
        self.min_slider.setValue(0.0)
        self.min_slider.setSingleStep(0.01)
        self.min_slider.setDecimals(2)
        self.min_slider.valueChanged.connect(self.on_min_changed)
        min_layout.addWidget(self.min_slider)
        control_layout.addLayout(min_layout)
        
        # Max distance
        max_layout = QVBoxLayout()
        max_layout.addWidget(QLabel("Maximum Distance (m):"))
        self.max_slider = QDoubleSpinBox()
        self.max_slider.setMinimum(0.0)
        self.max_slider.setMaximum(10.0)
        self.max_slider.setValue(10.0)
        self.max_slider.setSingleStep(0.01)
        self.max_slider.setDecimals(2)
        self.max_slider.valueChanged.connect(self.on_max_changed)
        max_layout.addWidget(self.max_slider)
        control_layout.addLayout(max_layout)
        
        # Statistics
        control_layout.addWidget(QLabel("\n--- Statistics ---"))
        self.stats_label = QLabel("")
        control_layout.addWidget(self.stats_label)
        
        # TOA preview
        control_layout.addWidget(QLabel("\n--- TOA Range (ms) ---"))
        self.toa_label = QLabel("")
        control_layout.addWidget(self.toa_label)
        
        control_layout.addStretch()
        
        # Buttons
        button_layout = QHBoxLayout()
        ok_btn = QPushButton("Validate")
        ok_btn.clicked.connect(self.accept)
        discard_btn = QPushButton("Discard")
        discard_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
        discard_btn.clicked.connect(self.on_discard)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(discard_btn)
        button_layout.addWidget(cancel_btn)
        control_layout.addLayout(button_layout)
        
        control_group.setLayout(control_layout)
        content_layout.addWidget(control_group, 1)
        
        layout.addLayout(content_layout)
        self.setLayout(layout)
        
        # Update statistics
        self.update_statistics()
    
    def on_discard(self):
        """Discard file and remove from dataset"""
        reply = QMessageBox.warning(
            self, "Discard File",
            f"Are you sure you want to discard this file?\n\n{self.filename}\n\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            # Emit signal to indicate file should be discarded
            self.file_discarded.emit(self.filename)
            # Close dialog
            self.reject()
    
    def on_min_changed(self, value):
        """Handle minimum distance change"""
        self.min_length = value
        if self.min_length > self.max_length:
            self.max_slider.blockSignals(True)
            self.max_slider.setValue(self.min_length)
            self.max_slider.blockSignals(False)
        t_min = self.min_length / self.C
        # Line is horizontal now (angle=0 for horizontal lines)
        self.line_min.setValue(t_min)
        # Re-find peaks and zoom heatmap
        self.reprocess_peaks_and_zoom()
    
    def on_max_changed(self, value):
        """Handle maximum distance change"""
        self.max_length = value
        if self.max_length < self.min_length:
            self.min_slider.blockSignals(True)
            self.min_slider.setValue(self.max_length)
            self.min_slider.blockSignals(False)
        t_max = self.max_length / self.C
        # Line is horizontal now (angle=0 for horizontal lines)
        self.line_max.setValue(t_max)
        # Re-find peaks and zoom heatmap
        self.reprocess_peaks_and_zoom()
    
    def reprocess_peaks_and_zoom(self):
        """Re-find peaks within current distance limits and zoom heatmap"""
        # Convert distance limits to time limits
        t_min = self.min_length / self.C
        t_max = self.max_length / self.C
        
        # Find indices in time array corresponding to limits
        idx_min = np.searchsorted(self.tt_limited, t_min)
        idx_max = np.searchsorted(self.tt_limited, t_max)
        
        # Re-find peaks for each microphone within the constrained range
        for i in range(self.gcce_limited.shape[0]):
            # Get GCCE for this microphone within the range
            gcce_range = self.gcce_limited[i, idx_min:idx_max]
            
            if len(gcce_range) > 0:
                # Find all peaks in this range
                peaks, properties = sig.find_peaks(gcce_range, height=0)
                
                if len(peaks) > 0:
                    # Get peak heights
                    peak_heights = properties['peak_heights']
                    
                    # Sort by height (descending) and get indices
                    sorted_indices = np.argsort(peak_heights)[::-1]
                    
                    # Get top 10 peaks (or fewer if fewer than 10 exist)
                    top_peaks = peaks[sorted_indices[:min(10, len(peaks))]]
                    
                    # Take the first (earliest) peak among the 10 highest
                    peak_in_range = np.min(top_peaks)
                    
                    # Convert back to global index
                    self.imax[i] = idx_min + peak_in_range
                else:
                    # No peaks in range, use closest to t_min
                    self.imax[i] = idx_min
            else:
                # Empty range, use t_min
                self.imax[i] = idx_min
        
        # Recompute TOA values based on new peaks
        self.toas = np.array([self.tt_limited[min(i, len(self.tt_limited)-1)] for i in self.imax])
        
        # Update peak markers
        self.update_peak_markers()
        
        # Zoom heatmap to new time range (with small margin for visibility)
        margin = 0.02 * (t_max - t_min)  # 2% margin
        self.plot_widget.setYRange(max(0, t_min - margin), min(self.tt_limited.max(), t_max + margin))
        
        # Update statistics with new peaks
        self.update_statistics()
    
    
    def update_peak_markers(self):
        """Update red dot markers for detected peaks (rotated coordinates)"""
        spots = []
        for mic_idx in range(self.gcce_limited.shape[0]):
            if self.imax[mic_idx] < len(self.tt_limited):
                t_peak = self.tt_limited[self.imax[mic_idx]]
                # Rotated: mic_idx on x-axis, time on y-axis
                spots.append({'pos': (mic_idx, t_peak), 'size': 4, 'pen': pg.mkPen(None), 'brush': pg.mkBrush('r')})
        self.peak_scatter.setData(spots)
    
    def update_statistics(self):
        """Update and display statistics for current range"""
        # Time bounds
        t_min = self.min_length / self.C
        t_max = self.max_length / self.C
        
        # Count valid peaks in range
        valid_peaks = sum(1 for t in self.toas if t_min <= t <= t_max)
        
        stats_text = (
            f"Distance: {self.min_length:.2f} - {self.max_length:.2f} m\n"
            f"Time: {t_min*1000:.3f} - {t_max*1000:.3f} ms\n"
            f"Valid Peaks: {valid_peaks}/{len(self.imax)}"
        )
        self.stats_label.setText(stats_text)
        
        # Compute TOA in selected range
        toas_in_range = []
        for mic_idx in range(len(self.imax)):
            if self.imax[mic_idx] < len(self.tt_limited):
                t_peak = self.tt_limited[self.imax[mic_idx]]
                if t_min <= t_peak <= t_max:
                    toas_in_range.append(t_peak)
                elif t_peak < t_min:
                    toas_in_range.append(t_min)
                else:
                    toas_in_range.append(t_max)
        
        if len(toas_in_range) > 0:
            toas_in_range = np.array(toas_in_range)
            toa_min = toas_in_range.min()
            toa_max = toas_in_range.max()
            toa_text = f"Min: {toa_min*1000:.3f} ms\nMax: {toa_max*1000:.3f} ms\nMean: {toas_in_range.mean()*1000:.3f} ms"
        else:
            toa_text = "No peaks in range"
        
        self.toa_label.setText(toa_text)
    
    def update_distance_axis(self):
        """Update secondary Y-axis to show distance in meters"""
        try:
            # Get current Y-axis range (time values)
            vb = self.plot_widget.getViewBox()
            if vb is None:
                return
            
            y_min, y_max = vb.viewRange()[1]
            
            # Convert time range to distance range
            d_min = self.C * y_min
            d_max = self.C * y_max
            
            # Create custom tick values: use time values that correspond to nice distance values
            num_ticks = 5
            # Create distance ticks first (nice values)
            distance_ticks = np.linspace(d_min, d_max, num_ticks)
            # Convert back to time positions
            time_positions = distance_ticks / self.C
            
            # Create tick labels showing distance
            tick_strings = [f"{d:.2f}" for d in distance_ticks]
            # Ticks format: (position_in_data_coords, label_text)
            ticks = [(time_positions[i], tick_strings[i]) for i in range(len(distance_ticks))]
            
            # Update right axis ticks
            self.right_axis.setTicks([ticks])
        except Exception as e:
            # Silently ignore errors during axis update
            pass
    
    def get_selected_range(self):
        """Return selected min/max distances in meters"""
        return self.min_length, self.max_length


class ProcessingThread(QThread):
    """
    Background thread for processing TOA (Time-of-Arrival) files.
    
    Supports both HDF5 and DAT+LOG file formats. Emits signals for progress tracking
    and individual map visualization. Implements GCC-PHAT cross-correlation for TOA extraction.
    Pauses at each file for GCC validation via dialog.
    """
    progress = pyqtSignal(int)  # Overall progress (0-100)
    file_progress = pyqtSignal(str, int)  # Emit filename and per-file progress (0-100)
    status = pyqtSignal(str)
    toa_map_ready = pyqtSignal(str, np.ndarray)  # Emit filename and TOA map as processed
    gcc_ready = pyqtSignal(str, object)  # Emit filename and GCC data tuple (GCCE, tt_limited, toas, imax, Tc, C)
    finished = pyqtSignal(list, list)  # Emit list of filenames and list of TOA maps
    error = pyqtSignal(str)
    
    def __init__(self, file_paths, temp_celsius=26, parent_window=None):
        super().__init__()
        self.file_paths = file_paths
        self.Tc = temp_celsius
        self.C = np.sqrt(1.4 * 287 * (self.Tc + 273))
        self.Fe = 50e3
        self.toa_maps = []
        self.filenames = []
        self.discarded_files = set()  # Track discarded files to exclude from final output
        self.parent_window = parent_window
        self.gcc_validated_range = None
        # Event to block thread until GCC validation is complete
        self.gcc_validation_event = threading.Event()
        self.gcc_validation_event.set()  # Initially set (not blocking)
        # Current processing file for progress tracking
        self.current_filename = None
        
    def run(self):
        try:
            total_files = len(self.file_paths)
            
            for idx, file_path in enumerate(self.file_paths):
                filename = os.path.basename(file_path)
                self.current_filename = filename
                
                self.status.emit(f"Processing: {filename}")
                self.file_progress.emit(filename, 10)  # File loading
                
                try:
                    # Always recompute TOA (no caching)
                    toa = self._process_single_file(file_path)
                    
                    if toa is not None:
                        self.file_progress.emit(filename, 90)  # Processing complete, pending validation
                        self.toa_maps.append(toa)
                        self.filenames.append(filename)
                        # Emit each map as processed for real-time visualization
                        self.toa_map_ready.emit(filename, toa)
                        
                        # Wait for GCC validation to complete before processing next file
                        self.gcc_validation_event.clear()
                        self.file_progress.emit(filename, 95)  # Waiting for validation
                        self.gcc_validation_event.wait(timeout=600)  # 10 minute timeout
                        self.file_progress.emit(filename, 100)  # Validation complete
                        
                except Exception as e:
                    self.status.emit(f"Skipped {filename}: {str(e)}")
                    self.file_progress.emit(filename, 0)  # Reset on error
                    continue
                
                progress = int((idx + 1) / total_files * 100)
                self.progress.emit(progress)
            
            if len(self.toa_maps) == 0:
                self.error.emit("No valid TOA data could be processed")
                return
            
            # Filter out discarded files before emitting final result
            final_filenames = []
            final_toa_maps = []
            for filename, toa_map in zip(self.filenames, self.toa_maps):
                if filename not in self.discarded_files:
                    final_filenames.append(filename)
                    final_toa_maps.append(toa_map)
            
            if len(final_toa_maps) == 0:
                self.error.emit("All processed files were discarded")
                return
            
            self.status.emit(f"Successfully processed {len(final_toa_maps)} files")
            self.finished.emit(final_filenames, final_toa_maps)
            
        except Exception as e:
            self.error.emit(f"Processing failed: {str(e)}")
    
    def _process_single_file(self, file_path):
        """Extract and process TOA from a single calibration file (HDF5 or DAT)"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Handle DAT files
        if file_ext == '.dat':
            return self._process_dat_file(file_path)
        
        # Handle HDF5 files
        elif file_ext in ['.h5', '.hdf5']:
            return self._process_h5_file(file_path)
        
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def _process_h5_file(self, file_path):
        """Process HDF5 calibration file"""
        data = h5py.File(file_path, 'r')
        filename = os.path.basename(file_path)
        
        try:
            Secs = [int(i) for i in data['muh5'].keys()]
            NbSecs = len(data['muh5'].keys())
            
            Sig = np.concatenate([data['muh5'][str(i)]['sig'][:] for i in range(NbSecs)], axis=1)
            
            Cmptr = Sig[0, :]
            NbTixels = len(Cmptr)
            
            # Check counter validity
            if np.sum(np.diff(Cmptr)) != NbTixels - 1:
                return None
            
            Mics = Sig[np.arange(1, 257), :]
            #Mics = np.vstack((Mics, Sig[259, :]))
            Ref = -Sig[257, :]  # Reference signal in Volts
            
            return self._compute_toa(Mics, Ref, NbTixels, filename=filename)
            
        finally:
            data.close()
    
    def _process_dat_file(self, file_path):
        """Process DAT calibration file"""
        if not DATPARSER_AVAILABLE:
            raise ImportError("DATParser not available. Install from rcbox package.")
        
        log_file = file_path.replace('.dat', '.log')
        if not os.path.exists(log_file):
            raise FileNotFoundError(f"Log file not found: {log_file}")
        
        filename = os.path.basename(file_path)
        
        try:
            dat_parser = DATParser(file_path, log_file)
            
            # Extract microphone signals
            Mics = dat_parser.extract_microphone_signals()
            
            # Extract analog signals (use first analog channel as reference if available)
            vas = dat_parser.extract_analog_signals()
            if vas.size > 0:
                Ref = vas[:, 0].astype(float)
            else:
                # Fallback: use first microphone as reference
                Ref = Mics[:, 0].astype(float)
            
            NbTixels = Mics.shape[0]
            
            return self._compute_toa(Mics, Ref, NbTixels, filename=filename)
            
        except Exception as e:
            raise RuntimeError(f"Error processing DAT file: {str(e)}")
    
    def _compute_toa(self, Mics, Ref, NbTixels, filename=None):
        """Compute TOA using GCC-PHAT algorithm with interactive GCC validation"""
        # GCC-PHAT processing
        NFFT = NbTixels
        df = self.Fe / NFFT
        tmax = 10.0 / self.C  # Fixed to 10 meters for validation
        
        SRef = np.fft.rfft(Ref, NFFT)
        SMics = np.fft.rfft(Mics, NFFT)
        GCS = np.conj(SRef)[None, :] * (SMics) / (np.abs(SMics) * np.abs(SRef)[None, :])
        
        NUp = 10 * NFFT
        dtUp = 1 / (NUp * df)
        GCC = np.fft.irfft(GCS, NUp)
        tt = np.arange(NUp) * dtUp
        
        # Limit to 10 meters and compute envelope
        valid_idx = tt < tmax
        GCCE = np.abs(sig.hilbert(GCC[:, valid_idx], axis=1))
        tt_limited = tt[valid_idx]
        
        # Find TOAs using peaks: find all peaks, take 10 highest, use earliest
        imax = np.zeros(GCCE.shape[0], dtype=int)
        for i in range(GCCE.shape[0]):
            # Find all peaks in this microphone's GCCE
            peaks, properties = sig.find_peaks(GCCE[i, :], height=0)
            
            if len(peaks) > 0:
                # Get peak heights
                peak_heights = properties['peak_heights']
                
                # Sort by height (descending) and get indices
                sorted_indices = np.argsort(peak_heights)[::-1]
                
                # Get top 10 peaks (or fewer if fewer than 10 exist)
                top_peaks = peaks[sorted_indices[:min(10, len(peaks))]]
                
                # Take the first (earliest) peak among the 10 highest
                imax[i] = np.min(top_peaks)
            else:
                # Fallback: use argmax if no peaks found
                imax[i] = np.argmax(GCCE[i, :])
        
        # Get initial TOA values
        toas = np.array([tt_limited[min(i, len(tt_limited)-1)] for i in imax])
        
        # Emit GCC data to main thread for dialog display
        # Use default full range (0-10m)
        if filename is not None:
            gcc_data = (GCCE, tt_limited, toas, imax, self.Tc, self.C)
            self.gcc_ready.emit(filename, gcc_data)
            self.gcc_validated_range = (0.0, 10.0)
        
        return toas
    
    def _compute_dtoa(self, toas, ref_mic_idx=0):
        """Compute Differential TOA (DTOA) matrix from TOA measurements.
        
        DTOA[i,j] = TOA[i] - TOA[j] for all microphone pairs (no reference channel)
        
        Parameters:
            toas: (NbMics,) array of TOA values for all mics
            ref_mic_idx: Index of reference microphone (not used for matrix form)
        
        Returns:
            dtoa_matrix: (NbMics, NbMics) 2D differential TOA matrix
                        Properties:
                        - Diagonal = 0 (mic i vs itself)
                        - Antisymmetric: dtoa[i,j] = -dtoa[j,i]
        
        Note: Each data file produces one such matrix.
              Multiple files assembled into (NbMics, NbMics, NbDatafiles) 3D array.
        """
        if len(toas) == 0:
            return np.array([])
        
        # Create 2D matrix: dtoa[i,j] = toa[i] - toa[j]
        toas_col = toas.reshape(-1, 1)  # Shape: (M, 1)
        toas_row = toas.reshape(1, -1)  # Shape: (1, M)
        dtoa_matrix = toas_col - toas_row  # Broadcasting creates (M, M)
        
        return dtoa_matrix


class SolverThread(QThread):
    """Background thread for rcbox RMDU (Reduced-Dimension Microdolly Units) solver.
    
    Runs iterative geometry optimization from DTOA (Differential TOA) measurements.
    Emits convergence history and iteration results for real-time visualization.
    
    Parameters:
        dtoa_matrix: (NbMics, NbMics, NbDatafiles) - DTOA for each data file (no reference channel)
        initial_xyz: Initial geometry guess (3, NbMics) or None for geometric initialization
        lambda_param: Damping factor for solver (0.001-0.1, default 0.05)
        max_iterations: Maximum optimization steps (default 10000)
        epsilon: Convergence tolerance (default 1e-10)
    """
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    iteration = pyqtSignal(np.ndarray)  # Emit XYZ at each iteration
    finished = pyqtSignal(dict)  # Return results dictionary
    error = pyqtSignal(str)
    
    def __init__(self, toa_matrix, initial_xyz=None, lambda_param=0.050, 
                 max_iter=10000, eps_limit=1e-10, temp_celsius=26, align_8=False):
        super().__init__()
        # Input: TOA matrix with shape (NbFiles, NbMics) or (NbMics, NbMics) if 2D
        # TOA values are in seconds, need to be converted to distances via multiplication by C
        
        self.align_8 = align_8
        self.toa_matrix = toa_matrix  # Store original for reference
        
        # Handle both 2D (single file) and 3D (multiple files) TOA arrays
        if toa_matrix.ndim == 3:
            # Multiple files: (NbFiles, NbMics, ...) - Average across files
            self.toa_matrix_avg = np.mean(toa_matrix[:, :, 0] if toa_matrix.shape[2] == 1 else toa_matrix, axis=0)
        elif toa_matrix.ndim == 2 and toa_matrix.shape[0] > toa_matrix.shape[1]:
            # Multiple files: (NbFiles, NbMics) - Average across files
            self.toa_matrix_avg = np.mean(toa_matrix, axis=0)
        else:
            # Single file or already averaged: just one row
            self.toa_matrix_avg = toa_matrix if toa_matrix.ndim == 1 else toa_matrix.reshape(-1)
        
        self.initial_xyz = initial_xyz
        self.lambda_param = lambda_param
        self.max_iter = max_iter
        self.eps_limit = eps_limit
        self.Tc = temp_celsius
        self.C = np.sqrt(1.4 * 287 * (self.Tc + 273))
        
    def run(self):
        """Run the RMDU solver in the background thread (Python implementation)."""
        try:
            if not RCBOX_AVAILABLE:
                self.error.emit("rcbox package not available")
                return
            
            self.status.emit("Initializing RMDU solver (Python)...")
            
            # 1. Prepare Distance Matrix
            # self.toa_matrix is (Ns, Nm) where Ns=NbFiles, Nm=NbMics
            # Distance = C * TOA
            # We want D_input to be (Nm, Ns)
            D_input = (self.C * self.toa_matrix).T 
            
            Nm, Ns = D_input.shape
            Nr = Nm + Ns # Total points = Sources + Mics
            
            # Construct full square EDM with Sources first (0..Ns-1) and Mics second (Ns..Nr-1)
            # D_full: (Nr, Nr)
            D_full = np.zeros((Nr, Nr))
            
            # Fill off-diagonal blocks
            # D_full[Ns:, :Ns] corresponds to Rows=Mics, Cols=Sources
            D_full[Ns:, :Ns] = D_input
            D_full[:Ns, Ns:] = D_input.T
            
            # Construct Weights
            # 1 for measured blocks, 0 for unknown blocks (S-S, M-M)
            W_full = np.zeros((Nr, Nr))
            W_full[Ns:, :Ns] = 1
            W_full[:Ns, Ns:] = 1
            
            Dflat = squareform(D_full)
            Wflat = squareform(W_full)
            
            # 2. Solver Parameters
            Ndim = 3
            Maxit = self.max_iter
            lbda = self.lambda_param
            
            # 3. Initialize X
            # Shape (Maxit, Ndim, Nr)
            X = np.zeros((Maxit, Ndim, Nr))
            if self.initial_xyz is None:
                X[0, :, :] = np.random.randn(Ndim, Nr)
            else:
                # If loaded geometry matches dimensions
                if self.initial_xyz.shape == (3, Nr):
                    X[0, :, :] = self.initial_xyz
                elif self.initial_xyz.shape == (Nr, 3):
                    X[0, :, :] = self.initial_xyz.T
                else:
                    X[0, :, :] = np.random.randn(Ndim, Nr)

            # 4. Precompute Lpinv
            Lpinv = compute_Lpinv(Nr, W_full)
            Eps = np.zeros(Maxit,)
            
            self.status.emit("Running RMDU solver...")
            
            t_final = 0
            for t in range(Maxit-1):
                # --- Alignment Step (Indices relative to X) ---
                if self.align_8:
                    current_X = X[t, :, :].copy()
                    
                    # Align chunks of 8 mics starting at Ns
                    # "starting from Ns in indices"
                    for i in range(Ns, Nr, 8):
                        # Chunk indices
                        chunk = np.arange(i, min(i+8, Nr))
                        if len(chunk) < 3: continue # Need at least 3 points for line fitting
                        
                        points = current_X[:, chunk]
                        # Fit line
                        centroid = np.mean(points, axis=1, keepdims=True)
                        centered = points - centroid
                        
                        try:
                            # SVD for line fitting
                            u, s, vh = linalg.svd(centered, full_matrices=False)
                            direction = u[:, 0:1] # Principal axis
                            
                            # Protect (Project points onto line)
                            proj = centroid + direction @ (direction.T @ centered)
                            current_X[:, chunk] = proj
                        except linalg.LinAlgError:
                            pass
                    
                    # Apply constrained X to current iteration
                    X[t, :, :] = current_X

                # --- Standard RMDU Step ---
                if t % 50 == 0:
                    self.iteration.emit(X[t, :, :].T)
                    self.progress.emit(int((t+1)/Maxit * 100))
                
                DDt = pdist(X[t, :, :].T)
                
                # O update: O[t+1] = S_lambda(W*(D-D_x), lambda)
                # Difference vector
                WeightedDiff = Wflat * (Dflat - DDt)
                O_val = np.sign(WeightedDiff) * np.maximum(np.abs(WeightedDiff) - lbda/2, 0.0)
                
                # Compute L1 (A11)
                # A11 = W * (D - O) / D_x
                with np.errstate(divide='ignore', invalid='ignore'):
                    term = Wflat * (Dflat - O_val) / DDt
                    A11 = np.where((DDt > 1e-9) & (Dflat > O_val), term, 0)
                
                A1 = squareform(A11)
                L1 = np.diag(A1.sum(1)) - A1
                
                # Update X
                # X_new = X_old * L1 * Lpinv
                X[t+1, :, :] = np.dot(X[t, :, :], np.dot(L1, Lpinv))
                
                # Convergence
                X_diff_norm = linalg.norm(X[t+1, :, :] - X[t, :, :])
                X_curr_norm = linalg.norm(X[t+1, :, :])
                if X_curr_norm > 1e-12:
                    Eps[t] = X_diff_norm / X_curr_norm
                else:
                    Eps[t] = 0
                
                if Eps[t] < self.eps_limit:
                    t_final = t
                    break
                t_final = t
            
            # --- Results ---
            XYZ_iters = X[0:t_final+1, :, :].transpose(0, 2, 1)
            XYZ_last = XYZ_iters[-1, ...].copy()
            
            results = {
                'XYZ_iters': XYZ_iters,
                'XYZ_final': XYZ_last,
                'Nm': Nr, # Reporting total points
                'Ns': Ns, # Distinct count
                'final_error': Eps[t_final]
            }
            
            self.status.emit(f"Solver completed in {t_final+1} iterations")
            self.finished.emit(results)
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(f"Solver failed: {str(e)}")


class TOAMapViewer(QDialog):
    """
    Interactive dialog for viewing and selecting processed TOA maps.
    
    Displays heatmap previews of Time-of-Arrival measurements from each file.
    Allows users to selectively include/exclude maps for geometry solving.
    Provides real-time visualization of TOA data quality.
    
    Features:
        • Thumbnail previews of TOA heatmaps
        • Checkbox selection (include/exclude files)
        • Select All / Deselect All buttons
        • Quality assessment via map preview
    """
    
    def __init__(self, filenames, toa_maps, parent=None):
        super().__init__(parent)
        self.filenames = filenames
        self.toa_maps = toa_maps
        self.selected_indices = []
        self.init_ui()
        self.setWindowTitle("TOA Maps - Select Maps to Process")
        self.setGeometry(100, 100, 1200, 600)
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Info label
        info = QLabel(f"Processed {len(self.filenames)} files. Select maps to include in geometry analysis:")
        layout.addWidget(info)
        
        # Horizontal layout for map display and list
        h_layout = QHBoxLayout()
        
        # Left: List with checkboxes
        list_layout = QVBoxLayout()
        list_label = QLabel("Files:")
        list_layout.addWidget(list_label)
        
        self.file_list = QListWidget()
        self.checkboxes = []
        
        for idx, filename in enumerate(self.filenames):
            item = QListWidgetItem(f"{idx+1}. {filename}")
            self.file_list.addItem(item)
            
            # Create checkbox widget
            checkbox = QCheckBox()
            checkbox.setChecked(True)  # Default to selected
            checkbox.stateChanged.connect(lambda state, i=idx: self.on_selection_changed(i, state))
            self.checkboxes.append(checkbox)
            self.file_list.setItemWidget(item, checkbox)
        
        list_layout.addWidget(self.file_list)
        h_layout.addLayout(list_layout, 1)
        
        # Right: Current TOA map visualization
        viz_layout = QVBoxLayout()
        viz_label = QLabel("TOA Map Preview:")
        viz_layout.addWidget(viz_label)
        
        self.map_viewer = pg.PlotWidget()
        self.map_viewer.setLabel('bottom', 'Microphone Index')
        self.map_viewer.setLabel('left', 'TOA (seconds)')
        self.map_viewer.setTitle('TOA Values per Microphone')
        # Plot initial TOA values
        mics = np.arange(len(self.toa_maps[0]))
        self.map_viewer.plot(mics, self.toa_maps[0], pen='b', symbol='o')
        viz_layout.addWidget(self.map_viewer)
        
        h_layout.addLayout(viz_layout, 2)
        layout.addLayout(h_layout)
        
        # Button layout
        button_layout = QHBoxLayout()
        
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all)
        button_layout.addWidget(select_all_btn)
        
        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(self.deselect_all)
        button_layout.addWidget(deselect_all_btn)
        
        button_layout.addStretch()
        
        ok_btn = QPushButton("Use Selected Maps")
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
        # Connect list selection to preview
        self.file_list.itemSelectionChanged.connect(self.on_file_selected)
    
    def on_file_selected(self):
        """Update preview when file is selected"""
        current_row = self.file_list.currentRow()
        if current_row >= 0 and current_row < len(self.toa_maps):
            # Clear plot and display new TOA values
            self.map_viewer.clear()
            mics = np.arange(len(self.toa_maps[current_row]))
            self.map_viewer.plot(mics, self.toa_maps[current_row], pen='b', symbol='o')
    
    def on_selection_changed(self, index, state):
        """Track selection state and update status"""
        num_selected = sum(1 for cb in self.checkboxes if cb.isChecked())
        total = len(self.checkboxes)
        # Note: self.toa_status only exists in GeoCalibApp, not in TOAMapViewer
        # So we'll skip updating status in this dialog
    
    def select_all(self):
        """Select all maps"""
        for checkbox in self.checkboxes:
            checkbox.setChecked(True)
    
    def deselect_all(self):
        """Deselect all maps"""
        for checkbox in self.checkboxes:
            checkbox.setChecked(False)
    
    def get_selected_maps(self):
        """Return selected TOA maps and filenames"""
        selected_filenames = []
        selected_maps = []
        
        for idx, checkbox in enumerate(self.checkboxes):
            if checkbox.isChecked():
                selected_filenames.append(self.filenames[idx])
                selected_maps.append(self.toa_maps[idx])
        
        return selected_filenames, selected_maps


class DTOAMapViewer(QDialog):
    """
    Interactive dialog for viewing and validating Differential TOA (DTOA) matrices.
    
    Second validation step: After TOA extraction and before geometry solving.
    Displays heatmap of DTOA values (differential from reference mic) for each file.
    Allows users to selectively include/exclude files based on DTOA quality.
    
    Features:
        • DTOA heatmap preview for each file
        • Quality assessment via heatmap patterns
        • Checkbox selection (include/exclude files)
        • Select All / Deselect All buttons
        • Statistics display (min, max, mean DTOA)
    """
    
    def __init__(self, filenames, dtoa_matrices, parent=None):
        super().__init__(parent)
        self.filenames = filenames
        self.dtoa_matrices = dtoa_matrices
        self.checkboxes = []
        self.setWindowTitle("DTOA Validation - Select Quality Data")
        self.setGeometry(150, 150, 1200, 700)
        self.init_ui()
    
    def init_ui(self):
        """Initialize DTOA viewer interface"""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Step 1B: Validate DTOA Matrices")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Info label
        info = QLabel(f"Review {len(self.filenames)} DTOA matrices. Check files to include in geometry solver.")
        layout.addWidget(info)
        
        # Main content: File list + Preview
        content_layout = QHBoxLayout()
        
        # Left: File list with checkboxes
        list_group = QGroupBox("DTOA Files")
        list_layout = QVBoxLayout()
        self.file_list = QListWidget()
        list_layout.addWidget(self.file_list)
        list_group.setLayout(list_layout)
        content_layout.addWidget(list_group, 1)
        
        # Right: DTOA heatmap preview
        preview_group = QGroupBox("DTOA Heatmap Preview")
        preview_layout = QVBoxLayout()
        self.preview_canvas = pg.ImageView()
        self.preview_canvas.setMinimumWidth(500)
        preview_layout.addWidget(self.preview_canvas)
        preview_group.setLayout(preview_layout)
        content_layout.addWidget(preview_group, 2)
        
        layout.addLayout(content_layout)
        
        # Statistics
        stats_layout = QHBoxLayout()
        self.stats_label = QLabel("Stats: ")
        stats_layout.addWidget(self.stats_label)
        stats_layout.addStretch()
        layout.addLayout(stats_layout)
        
        # Populate file list with checkboxes
        for idx, filename in enumerate(self.filenames):
            item = QListWidgetItem()
            item.setText(filename)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)  # Default: all selected
            self.file_list.addItem(item)
            self.checkboxes.append(item)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all)
        button_layout.addWidget(select_all_btn)
        
        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(self.deselect_all)
        button_layout.addWidget(deselect_all_btn)
        
        button_layout.addStretch()
        
        ok_btn = QPushButton("Use Selected DTOAs")
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
        # Connect list selection to preview
        self.file_list.itemSelectionChanged.connect(self.on_file_selected)
    
    def on_file_selected(self):
        """Display selected DTOA heatmap and statistics"""
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            return
        
        item = selected_items[0]
        idx = self.file_list.row(item)
        dtoa_matrix = self.dtoa_matrices[idx]
        
        # Display DTOA heatmap with inferno colormap
        self.preview_canvas.setImage(np.abs(dtoa_matrix), autoRange=True, autoLevels=True)
        
        # Apply inferno colormap
        cmap = pg.colormap.get('inferno')
        self.preview_canvas.imageItem.setColorMap(cmap)
        
        # Display statistics
        stats_text = (
            f"File: {self.filenames[idx]} | "
            f"Min: {np.min(dtoa_matrix):.4f} | "
            f"Max: {np.max(dtoa_matrix):.4f} | "
            f"Mean: {np.mean(dtoa_matrix):.4f} | "
            f"Std: {np.std(dtoa_matrix):.4f}"
        )
        self.stats_label.setText(stats_text)
    
    def select_all(self):
        """Check all files"""
        for item in self.checkboxes:
            item.setCheckState(Qt.Checked)
    
    def deselect_all(self):
        """Uncheck all files"""
        for item in self.checkboxes:
            item.setCheckState(Qt.Unchecked)
    
    def get_selected_dtoas(self):
        """Return selected filenames and DTOA matrices"""
        selected_filenames = []
        selected_dtoas = []
        
        for idx, item in enumerate(self.checkboxes):
            if item.checkState() == Qt.Checked:
                selected_filenames.append(self.filenames[idx])
                selected_dtoas.append(self.dtoa_matrices[idx])
        
        return selected_filenames, selected_dtoas


class AlignmentDialog(QDialog):
    """
    Dialog for input reference microphone positions for alignment.
    
    Allows users to specify known positions of reference microphones
    for Procrustes-based alignment of solver geometry. Used to transform
    computed positions to absolute coordinates.
    
    Parameters:
        num_ref_mics: Number of reference microphones to specify (default 4)
    """
    
    def __init__(self, parent=None, num_ref_mics=4):
        super().__init__(parent)
        self.num_ref_mics = num_ref_mics
        self.ref_positions = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Microphone Reference Positions")
        self.setGeometry(100, 100, 700, 500)
        
        layout = QVBoxLayout()
        
        # Table for positions
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Microphone", "X (mm)", "Y (mm)", "Z (mm)"])
        self.table.setRowCount(self.num_ref_mics)
        
        for i in range(self.num_ref_mics):
            item = QTableWidgetItem(f"Ref {i+1}")
            self.table.setItem(i, 0, item)
            for j in range(1, 4):
                self.table.setItem(i, j, QTableWidgetItem("0.0"))
        
        layout.addWidget(QLabel("Enter reference microphone positions (in mm):"))
        layout.addWidget(self.table)
        
        # Buttons
        button_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def get_positions(self):
        """Return positions in meters"""
        positions = []
        for i in range(self.num_ref_mics):
            try:
                x = float(self.table.item(i, 1).text()) * 1e-3
                y = float(self.table.item(i, 2).text()) * 1e-3
                z = float(self.table.item(i, 3).text()) * 1e-3
                positions.append([x, y, z])
            except ValueError:
                raise ValueError(f"Invalid value in row {i}")
        return np.array(positions)


class MicrophonePositionDialog(QDialog):
    """Dialog for selecting which microphones from array correspond to reference positions"""
    
    def __init__(self, parent=None, total_mics=257, num_ref=4):
        super().__init__(parent)
        self.total_mics = total_mics
        self.num_ref = num_ref
        self.mic_indices = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Select Microphone Indices")
        self.setGeometry(100, 100, 400, 300)
        
        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"Select {self.num_ref} microphone indices from array (0-{self.total_mics-1}):"))
        
        # Spinboxes for microphone indices
        self.spinboxes = []
        indices_layout = QFormLayout()
        for i in range(self.num_ref):
            spinbox = QSpinBox()
            spinbox.setMinimum(0)
            spinbox.setMaximum(self.total_mics - 1)
            spinbox.setValue(i * (self.total_mics // self.num_ref))
            indices_layout.addRow(f"Ref {i+1}:", spinbox)
            self.spinboxes.append(spinbox)
        
        layout.addLayout(indices_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def get_indices(self):
        """Return selected microphone indices"""
        return np.array([spinbox.value() for spinbox in self.spinboxes])


class GeoCalibApp(QMainWindow):
    """
    Main PyQt5 application for microphone array geometric calibration.
    
    Implements 3-tab workflow: TOA extraction → Geometry solving → Alignment & Export.
    Supports batch processing with real-time visualization and parameter tuning.
    
    GUI LAYOUT:
        Tab 1 (TOA):
            • File selection (HDF5 or DAT+LOG)
            • Temperature input → Sound speed calculation
            • Progress bar + Map viewer with checkbox selection
        
        Tab 2 (Solver):
            • RMDU parameters (lambda, iterations, epsilon)
            • Real-time 3D visualization of convergence
            • Iteration history plot
        
        Tab 3 (Alignment):
            • Reference position input (Procrustes alignment)
            • Manual rotation/translation
            • Export options (NPY, JSON, CSV)
    
    SIGNALS:
        • ProcessingThread: progress, status, toa_map_ready, finished, error
        • SolverThread: progress, status, iteration_ready, finished, error
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Geocalibration Processing App")
        self.setGeometry(100, 100, 1400, 900)
        
        # Data storage - TOA Validation (Step 1)
        self.toa_matrix = None
        self.toa_files = []
        self.selected_toa_filenames = []
        
        # Data storage - DTOA Validation (Step 1B)
        self.dtoa_matrices = []
        self.dtoa_files = []
        self.selected_dtoa_filenames = []
        self.dtoa_matrix = None  # Assembled DTOA matrix for solver
        
        # Data storage - Geometry Solving (Step 2)
        self.xyz_iters = None
        self.xyz_final = None
        self.xyz_init = None  # Initial geometry for solver (loaded from previous run)
        self.ref_positions = None
        self.mic_indices = None
        self.alignment_rotation = None
        self.alignment_translation = None
        
        # Parameters
        self.temperature = 26.0
        self.lambda_param = 0.050
        self.ref_mic_index = 0  # Reference microphone for DTOA computation
        self.processed_files = set()  # Track which files have been processed
        self.num_selected_sources = 0  # Store count at start of processing
        
        # Threads
        self.processing_thread = None
        self.solver_thread = None
        
        # Create menu bar
        self.create_menu_bar()
        # Animation
        self.animation_timer = None
        self.current_frame = 0
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        
        # Left panel: Controls
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Right panel: Visualization
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 2)
        
        central_widget.setLayout(main_layout)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
    def create_left_panel(self):
        """Create control panel on the left"""
        group = QGroupBox("Processing Controls")
        layout = QVBoxLayout()
        
        # Create tabs
        tabs = QTabWidget()
        
        # Tab 1: Step 1 - TOA Processing
        tab1 = QWidget()
        tab1_layout = QVBoxLayout()
        tab1_layout.addWidget(self.create_toa_processing_section())
        tab1.setLayout(tab1_layout)
        tabs.addTab(tab1, "Step 1: TOA Processing")
        
        # Tab 2: Step 2 - Geometry Inference
        tab2 = QWidget()
        tab2_layout = QVBoxLayout()
        tab2_layout.addWidget(self.create_geometry_section())
        tab2.setLayout(tab2_layout)
        tabs.addTab(tab2, "Step 2: Geometry")
        
        # Tab 3: Step 3 - Alignment
        tab3 = QWidget()
        tab3_layout = QVBoxLayout()
        tab3_layout.addWidget(self.create_alignment_section())
        tab3.setLayout(tab3_layout)
        tabs.addTab(tab3, "Step 3: Alignment")
        
        # Tab 4: Visualization Options
        tab4 = QWidget()
        tab4_layout = QVBoxLayout()
        tab4_layout.addWidget(self.create_visualization_section())
        tab4.setLayout(tab4_layout)
        tabs.addTab(tab4, "Visualization")
        
        layout.addWidget(tabs)
        group.setLayout(layout)
        return group
    
    def create_menu_bar(self):
        """Create application menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        save_toa_action = file_menu.addAction("Save TOA Matrix")
        save_toa_action.triggered.connect(self.save_toa_matrix)
        save_toa_action.setToolTip("Save the validated TOA matrix from Step 1")
        
        load_toa_action = file_menu.addAction("Load TOA Matrix (Step 1)")
        load_toa_action.triggered.connect(self.load_toa_matrix)
        load_toa_action.setToolTip("Load a previously saved TOA matrix in Step 1")
        
        load_toa_step2_action = file_menu.addAction("Load TOA Matrix (Step 2)")
        load_toa_step2_action.triggered.connect(self.load_toa_matrix_for_step2)
        load_toa_step2_action.setToolTip("Load a previously saved TOA matrix in Step 2")
        
        file_menu.addSeparator()
        
        save_geometry_action = file_menu.addAction("Save Geometry (XYZ)")
        save_geometry_action.triggered.connect(self.save_geometry)
        save_geometry_action.setToolTip("Save the final inferred geometry from Step 2")
        
        load_geometry_xinit_action = file_menu.addAction("Load Geometry as Xinit (Step 2)")
        load_geometry_xinit_action.triggered.connect(self.load_geometry_as_xinit)
        load_geometry_xinit_action.setToolTip("Load previously saved geometry as initial input for solver")
        
        file_menu.addSeparator()
        
        save_xyz_action = file_menu.addAction("Save XYZ Results")
        save_xyz_action.triggered.connect(self.save_xyz_results)
        
        save_config_action = file_menu.addAction("Save Configuration")
        save_config_action.triggered.connect(self.save_configuration)
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self.show_help)
    
    def show_help(self):
        """Show application information"""
        msg = ("GeoCalibApp - Geomicrophone Calibration Tool\n\n"
               "3-Step Workflow:\n"
               "1. TOA Processing: Extract Time-of-Arrival from GCC-PHAT\n"
               "2. TOA → Geo: Convert TOA differences to GPS coordinates\n"
               "3. Geo → Calib: Generate calibration data\n\n"
               "For detailed help, see USER_MANUAL.md")
        QMessageBox.information(self, "About", msg)
    
    def create_toa_processing_section(self):
        """Create TOA processing controls"""
        group = QGroupBox("Step 1: TOA Processing")
        layout = QVBoxLayout()
        
        # File selection
        file_layout = QHBoxLayout()
        self.file_list = QListWidget()
        file_layout.addWidget(self.file_list)
        layout.addWidget(QLabel("Selected Files:"))
        layout.addLayout(file_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        browse_btn = QPushButton("Browse Files")
        browse_btn.clicked.connect(self.browse_toa_files)
        clear_btn = QPushButton("Clear List")
        clear_btn.clicked.connect(self.clear_file_list)
        button_layout.addWidget(browse_btn)
        button_layout.addWidget(clear_btn)
        layout.addLayout(button_layout)
        
        # Parameters
        param_layout = QFormLayout()
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setValue(26.0)
        self.temp_spin.setSuffix(" °C")
        param_layout.addRow("Temperature:", self.temp_spin)
        layout.addLayout(param_layout)
        
        # Process button
        self.process_btn = QPushButton("Process TOAs from Calibration data files")
        self.process_btn.clicked.connect(self.process_toa)
        self.process_btn.setEnabled(False)  # Disabled until files are selected
        layout.addWidget(self.process_btn)
        
        # Progress
        layout.addWidget(QLabel("Overall Progress:"))
        self.toa_progress = QProgressBar()
        layout.addWidget(self.toa_progress)
        
        self.toa_status = QLabel("No sources selected")
        layout.addWidget(self.toa_status)
        
        # Save/Load buttons for validated TOA matrix
        layout.addWidget(QLabel("\n--- Validated Matrix Management ---"))
        toa_io_layout = QHBoxLayout()
        
        save_toa_btn = QPushButton("Save Validated TOA")
        save_toa_btn.clicked.connect(self.save_toa_matrix)
        save_toa_btn.setToolTip("Save the validated TOA matrix for later use")
        toa_io_layout.addWidget(save_toa_btn)
        
        load_toa_btn = QPushButton("Load Validated TOA")
        load_toa_btn.clicked.connect(self.load_toa_matrix)
        load_toa_btn.setToolTip("Load a previously saved TOA matrix")
        toa_io_layout.addWidget(load_toa_btn)
        
        layout.addLayout(toa_io_layout)
        
        layout.addStretch()
        group.setLayout(layout)
        return group
    
    def create_geometry_section(self):
        """Create geometry inference controls"""
        group = QGroupBox("Step 2: Geometry Inference (rcbox RMDU)")
        layout = QVBoxLayout()
        
        if not RCBOX_AVAILABLE:
            layout.addWidget(QLabel("⚠ rcbox package not available. Install with: pip install rcbox"))
        
        # Load TOA matrix section
        layout.addWidget(QLabel("--- Load TOA Matrix ---"))
        load_geometry_layout = QHBoxLayout()
        
        load_toa_step2_btn = QPushButton("Load TOA Matrix")
        load_toa_step2_btn.clicked.connect(self.load_toa_matrix_for_step2)
        load_toa_step2_btn.setToolTip("Load a previously saved TOA matrix to skip Step 1")
        load_geometry_layout.addWidget(load_toa_step2_btn)
        
        self.loaded_toa_label = QLabel("No TOA matrix loaded")
        load_geometry_layout.addWidget(self.loaded_toa_label, 1)
        
        layout.addLayout(load_geometry_layout)
        
        # Parameters
        param_layout = QFormLayout()
        
        self.lambda_spin = QDoubleSpinBox()
        self.lambda_spin.setValue(0.050)
        self.lambda_spin.setDecimals(4)
        self.lambda_spin.setSingleStep(0.001)
        param_layout.addRow("Lambda (regularization):", self.lambda_spin)
        
        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setValue(10000)
        self.max_iter_spin.setMaximum(100000)
        param_layout.addRow("Max Iterations:", self.max_iter_spin)
        
        self.eps_spin = QDoubleSpinBox()
        self.eps_spin.setValue(1e-6)
        self.eps_spin.setDecimals(12)
        param_layout.addRow("Epsilon Limit:", self.eps_spin)

        self.align8_check = QCheckBox("Alignement par 8")
        self.align8_check.setChecked(False)
        self.align8_check.setToolTip("Constrain microphones to be aligned by chunks of 8")
        param_layout.addRow("", self.align8_check)
        
        layout.addLayout(param_layout)
        
        # Load geometry as Xinit section
        layout.addWidget(QLabel("--- Load Initial Geometry ---"))
        load_xinit_layout = QHBoxLayout()
        
        load_xinit_btn = QPushButton("Load Geometry as Xinit")
        load_xinit_btn.clicked.connect(self.load_geometry_as_xinit)
        load_xinit_btn.setToolTip("Load previously saved geometry as initial guess for solver")
        load_xinit_layout.addWidget(load_xinit_btn)
        
        self.loaded_xinit_label = QLabel("No initial geometry loaded")
        load_xinit_layout.addWidget(self.loaded_xinit_label, 1)
        
        layout.addLayout(load_xinit_layout)
        
        # Run solver button
        solver_btn = QPushButton("Run Solver")
        solver_btn.clicked.connect(self.run_solver)
        layout.addWidget(solver_btn)
        
        # Progress
        self.solver_progress = QProgressBar()
        layout.addWidget(self.solver_progress)
        
        self.solver_status = QLabel("No solver run")
        layout.addWidget(self.solver_status)
        
        # Results info
        self.solver_info = QLabel("")
        layout.addWidget(self.solver_info)
        
        layout.addStretch()
        group.setLayout(layout)
        return group
    
    def create_alignment_section(self):
        """Create antenna alignment controls"""
        group = QGroupBox("Step 3: Antenna Alignment")
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("Align antenna using reference microphone positions"))
        
        # Number of reference mics
        num_layout = QFormLayout()
        self.num_ref_spin = QSpinBox()
        self.num_ref_spin.setValue(4)
        self.num_ref_spin.setMinimum(3)
        self.num_ref_spin.setMaximum(16)
        num_layout.addRow("Number of Ref Mics:", self.num_ref_spin)
        layout.addLayout(num_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        ref_pos_btn = QPushButton("Enter Ref Positions")
        ref_pos_btn.clicked.connect(self.enter_reference_positions)
        button_layout.addWidget(ref_pos_btn)
        
        mic_idx_btn = QPushButton("Select Mic Indices")
        mic_idx_btn.clicked.connect(self.select_microphone_indices)
        button_layout.addWidget(mic_idx_btn)
        
        layout.addLayout(button_layout)
        
        # Align button
        align_btn = QPushButton("Perform Alignment")
        align_btn.clicked.connect(self.perform_alignment)
        layout.addWidget(align_btn)
        
        self.align_status = QLabel("No alignment performed")
        layout.addWidget(self.align_status)
        
        # Export results
        layout.addWidget(QLabel("Export Results:"))
        export_layout = QHBoxLayout()
        
        save_xyz_btn = QPushButton("Save XYZ")
        save_xyz_btn.clicked.connect(self.save_xyz_results)
        export_layout.addWidget(save_xyz_btn)
        
        save_config_btn = QPushButton("Save Config")
        save_config_btn.clicked.connect(self.save_configuration)
        export_layout.addWidget(save_config_btn)
        
        layout.addLayout(export_layout)
        
        layout.addStretch()
        group.setLayout(layout)
        return group
    
    def create_visualization_section(self):
        """Create visualization controls"""
        group = QGroupBox("Visualization & Animation")
        layout = QVBoxLayout()
        
        # Plot options
        plot_layout = QHBoxLayout()
        
        plot_toa_btn = QPushButton("Plot TOA Matrix")
        plot_toa_btn.clicked.connect(self.plot_toa_matrix)
        plot_layout.addWidget(plot_toa_btn)
        
        plot_final_btn = QPushButton("Plot Final Geometry")
        plot_final_btn.clicked.connect(self.plot_final_geometry)
        plot_layout.addWidget(plot_final_btn)
        
        layout.addLayout(plot_layout)
        
        # Animation controls
        layout.addWidget(QLabel("Iteration Animation:"))
        
        anim_layout = QHBoxLayout()
        
        self.animate_checkbox = QCheckBox("Enable Animation")
        anim_layout.addWidget(self.animate_checkbox)
        
        anim_speed = QLabel("Speed:")
        anim_layout.addWidget(anim_speed)
        
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(10)
        self.speed_slider.setMaximum(500)
        self.speed_slider.setValue(50)
        anim_layout.addWidget(self.speed_slider)
        
        layout.addLayout(anim_layout)
        
        # Frame counter
        frame_layout = QHBoxLayout()
        self.frame_label = QLabel("Frame: 0 / 0")
        frame_layout.addWidget(self.frame_label)
        layout.addLayout(frame_layout)
        
        # Play/Pause/Save
        control_layout = QHBoxLayout()
        
        play_btn = QPushButton("Play Animation")
        play_btn.clicked.connect(self.play_animation)
        control_layout.addWidget(play_btn)
        
        pause_btn = QPushButton("Pause")
        pause_btn.clicked.connect(self.pause_animation)
        control_layout.addWidget(pause_btn)
        
        save_anim_btn = QPushButton("Save Animation")
        save_anim_btn.clicked.connect(self.save_animation)
        control_layout.addWidget(save_anim_btn)
        
        layout.addLayout(control_layout)
        
        layout.addStretch()
        group.setLayout(layout)
        return group
    
    def create_right_panel(self):
        """Create 3D visualization panel on the right"""
        group = QGroupBox("3D Visualization")
        layout = QVBoxLayout()
        
        # 3D OpenGL view
        self.view_3d = gl.GLViewWidget()
        self.view_3d.opts['distance'] = 5
        
        # Add grid
        grid = gl.GLGridItem()
        self.view_3d.addItem(grid)
        
        # Axis
        axis = gl.GLAxisItem()
        axis.setSize(1, 1, 1)
        self.view_3d.addItem(axis)
        
        layout.addWidget(self.view_3d)
        group.setLayout(layout)
        return group
    
    # ==================== TOA Processing ====================
    def browse_toa_files(self):
        """Open file browser for TOA calibration files (HDF5 and DAT formats)"""
        file_dir, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Calibration Files (HDF5 or DAT)",
            "",
            "All Supported (*.h5 *.hdf5 *.dat);;HDF5 Files (*.h5 *.hdf5);;DAT Files (*.dat);;All Files (*)"
        )
        if file_dir:
            self.toa_files.extend(file_dir)
            for f in file_dir:
                self.file_list.addItem(os.path.basename(f))
            self.update_process_button_state()
            self.update_process_button_state()
    
    def clear_file_list(self):
        """Clear the file list"""
        self.file_list.clear()
        self.toa_files.clear()
        self.toa_status.setText("File list cleared")
        self.update_process_button_state()
        self.update_process_button_state()
    
    def update_process_button_state(self):
        """Enable/disable and highlight process button based on file selection"""
        has_files = len(self.toa_files) > 0
        self.process_btn.setEnabled(has_files)
        
        if has_files:
            # Highlight button with green background and bold text
            self.process_btn.setStyleSheet(
                "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 6px; }"
                "QPushButton:hover { background-color: #45a049; }"
                "QPushButton:pressed { background-color: #3d8b40; }"
            )
        else:
            # Reset to default styling
            self.process_btn.setStyleSheet("")
    
    def get_selected_sources_count(self):
        """Count number of selected sources (checked files)"""
        if not hasattr(self, 'checkboxes'):
            return 0
        count = 0
        for checkbox in self.checkboxes:
            if checkbox.isChecked():
                count += 1
        return count
    
    def process_toa(self):
        """Process TOA from calibration files"""
        if not self.toa_files:
            QMessageBox.warning(self, "Error", "No files selected")
            return
        
        self.temperature = self.temp_spin.value()
        
        # Store number of selected sources for progress tracking
        self.num_selected_sources = self.get_selected_sources_count()
        self.processed_files.clear()  # Reset processed files counter
        
        self.statusBar().showMessage("Processing TOA files...")
        
        self.processing_thread = ProcessingThread(
            self.toa_files,
            temp_celsius=self.temperature,
            parent_window=self
        )
        self.processing_thread.progress.connect(self.toa_progress.setValue)
        self.processing_thread.file_progress.connect(self.on_file_progress)
        self.processing_thread.status.connect(self.toa_status.setText)
        self.processing_thread.toa_map_ready.connect(self.on_toa_map_ready)
        self.processing_thread.gcc_ready.connect(self.on_gcc_ready)
        self.processing_thread.finished.connect(self.on_toa_processed)
        self.processing_thread.error.connect(self.on_processing_error)
        self.processing_thread.start()
    
    def on_file_progress(self, filename, progress):
        """Handle per-file progress updates"""
        # Mark file as processed when progress reaches 100%
        if progress == 100:
            self.processed_files.add(filename)
            # Update file list to show checkmark - search by filename in text
            for idx in range(self.file_list.count()):
                item = self.file_list.item(idx)
                if item and filename in item.text():
                    # Add checkmark at the beginning if not already present
                    text = item.text()
                    if not text.startswith("\u2713") and not text.startswith("\u274c"):
                        item.setText(f"\u2713 {text}")
                    break
            # Update status with processed source count
            num_processed = len(self.processed_files)
            self.toa_status.setText(f"Processed {num_processed}/{self.num_selected_sources} sources")
    
    def on_toa_map_ready(self, filename, toa_map):
        """Handle individual TOA map as processed (for real-time visualization)"""
        # Could update a live preview here if desired
        pass
    
    def on_gcc_ready(self, filename, gcc_data):
        """Handle GCC data ready for validation (called in main thread)"""
        GCCE, tt_limited, toas, imax, Tc, C = gcc_data
        
        # Show GCC validation dialog in main thread (SAFE!)
        dialog = GCCValidationDialog(
            filename,
            GCCE,
            tt_limited,
            toas,
            imax,
            temp_celsius=Tc,
            parent=self
        )
        
        # Connect discard signal to handler
        dialog.file_discarded.connect(self.on_file_discarded)
        
        result = dialog.exec_()
        
        if result == QDialog.Accepted:
            min_dist, max_dist = dialog.get_selected_range()
            self.statusBar().showMessage(f"GCC validated: {filename} ({min_dist:.2f}-{max_dist:.2f}m)")
        
        # Signal processing thread to continue (unblock it)
        if self.processing_thread is not None:
            self.processing_thread.gcc_validation_event.set()
    
    def on_file_discarded(self, filename):
        """Handle discarded file - mark as discarded to exclude from final results"""
        # Mark file as discarded (will be filtered out before final emission)
        if self.processing_thread is not None:
            self.processing_thread.discarded_files.add(filename)
            # DO NOT set the event here - let on_gcc_ready handle it after dialog closes
        
        # Remove from main window's toa_files list
        if filename in self.toa_files:
            self.toa_files.remove(filename)
        
        # Update file list UI to show it's discarded
        for idx in range(self.file_list.count()):
            item = self.file_list.item(idx)
            if item and filename in item.text():
                item.setText(f"\u274c {item.text()}")  # Add X symbol to discarded items
                break
        
        # Update status
        num_selected = self.get_selected_sources_count()
        num_processed = len(self.processed_files)
        self.toa_status.setText(f"Processed {num_processed}/{num_selected} sources (1 discarded)")
        
        self.statusBar().showMessage(f"File discarded: {filename}")
    
    def on_toa_processed(self, filenames, toa_maps):
        """Handle completed TOA processing and show map viewer"""
        if len(toa_maps) == 0:
            QMessageBox.warning(self, "Error", "No valid TOA maps processed")
            return
        
        # Step 1A: Show TOA map viewer dialog
        viewer = TOAMapViewer(filenames, toa_maps, self)
        result = viewer.exec_()
        
        if result == QDialog.Accepted:
            # Get selected maps
            selected_filenames, selected_maps = viewer.get_selected_maps()
            
            if len(selected_maps) == 0:
                QMessageBox.warning(self, "Error", "No maps selected")
                return
            
            # Convert to numpy array and store
            self.toa_matrix = np.array(selected_maps)
            self.selected_toa_filenames = selected_filenames
            
            msg = f"Using {len(selected_maps)}/{len(filenames)} sources. Matrix shape: {self.toa_matrix.shape}"
            self.toa_status.setText(msg)
            self.statusBar().showMessage(f"TOA Validation 1A complete: {msg}")
            
            # Step 1B: Compute DTOA and show validation dialog
            self.compute_and_validate_dtoa(selected_filenames, selected_maps)
    
    def _compute_dtoa(self, toas):
        """
        Compute Differential TOA (DTOA) matrix from TOA measurements.
        DTOA[i,j] = TOA[i] - TOA[j] for all microphone pairs
        
        Parameters:
            toas: (M,) array of TOA values for all mics
        
        Returns:
            dtoa_matrix: (M, M) 2D array of differential TOAs between all mic pairs
        """
        if len(toas) == 0:
            return np.array([])
        
        # Create 2D matrix: dtoa[i,j] = toa[i] - toa[j]
        toas_col = toas.reshape(-1, 1)  # Shape: (M, 1)
        toas_row = toas.reshape(1, -1)  # Shape: (1, M)
        dtoa_matrix = toas_col - toas_row  # Broadcasting creates (M, M)
        
        return dtoa_matrix
    
    def compute_and_validate_dtoa(self, filenames, toa_maps):
        """Compute DTOA for each file and show validation dialog"""
        dtoa_matrices = []
        
        # Compute DTOA matrix for each file (NbMics x NbMics difference matrix)
        for toa in toa_maps:
            # DTOA[i,j] = TOA[i] - TOA[j] for all microphone pairs
            dtoa_matrix = self._compute_dtoa(toa)
            dtoa_matrices.append(dtoa_matrix)
        
        # Store DTOA data
        self.dtoa_matrices = dtoa_matrices
        self.dtoa_files = filenames
        
        # Step 1B: Show DTOA validation dialog
        dtoa_viewer = DTOAMapViewer(filenames, dtoa_matrices, self)
        result = dtoa_viewer.exec_()
        
        if result == QDialog.Accepted:
            # Get selected DTOAs for reference/validation, but use TOA for solver
            selected_filenames, selected_dtoas = dtoa_viewer.get_selected_dtoas()
            
            if len(selected_dtoas) == 0:
                QMessageBox.warning(self, "Error", "No DTOA matrices selected")
                return
            
            # Filter TOA matrix to match selected files (same indices as selected DTOAs)
            # Get indices of selected files from the full filenames list
            all_filenames_in_matrix = self.dtoa_files  # The filenames list from compute_and_validate_dtoa
            selected_indices = [all_filenames_in_matrix.index(fname) for fname in selected_filenames]
            
            # Update toa_matrix to include only selected files (absolute TOA in seconds)
            original_toa_matrix = self.toa_matrix  # We stored this earlier from selected_maps
            filtered_toa_matrix = original_toa_matrix[selected_indices, :]
            self.toa_matrix = filtered_toa_matrix  # Shape: (NbSelectedFiles, NbMics) - absolute TOA
            self.selected_toa_filenames = selected_filenames
            self.selected_dtoa_filenames = selected_filenames
            
            # Store DTOA for reference only (visualization/diagnostics)
            self.dtoa_matrices = selected_dtoas
            
            msg = f"Using {len(selected_dtoas)}/{len(filenames)} sources. TOA matrix shape: {self.toa_matrix.shape}"
            self.toa_status.setText(msg)
            self.statusBar().showMessage(f"TOA/DTOA Validation complete: {msg}")
            QMessageBox.information(self, "Success", 
                f"TOA/DTOA validation complete\nReady for geometry inference solver\n{msg}")
    
    def on_processing_error(self, error_msg):
        """Handle processing error"""
        self.statusBar().showMessage("Error during processing")
        QMessageBox.critical(self, "Processing Error", error_msg)
    
    # ==================== Geometry Inference ====================
    def run_solver(self):
        """Run the rcbox RMDU solver"""
        if self.toa_matrix is None:
            QMessageBox.warning(self, "Error", "No TOA matrix available. Complete TOA validation first.")
            return
        
        if not RCBOX_AVAILABLE:
            QMessageBox.critical(self, "Error", "rcbox package not available. Install it first.")
            return
        
        self.temperature = self.temp_spin.value()
        self.lambda_param = self.lambda_spin.value()
        
        self.statusBar().showMessage("Running RMDU solver...")
        
        # Create solver with absolute TOA matrix (not DTOA)
        # Use loaded initial geometry if available
        initial_xyz = self.xyz_init if hasattr(self, 'xyz_init') and self.xyz_init is not None else None
        
        self.solver_thread = SolverThread(
            self.toa_matrix,  # Absolute TOA in seconds, shape: (NbFiles, NbMics)
            initial_xyz=initial_xyz,  # Use loaded geometry or None for from-scratch solver
            lambda_param=self.lambda_param,
            max_iter=self.max_iter_spin.value(),
            eps_limit=self.eps_spin.value(),
            temp_celsius=self.temperature,
            align_8=self.align8_check.isChecked()
        )
        self.solver_thread.progress.connect(self.solver_progress.setValue)
        self.solver_thread.status.connect(self.solver_status.setText)
        self.solver_thread.iteration.connect(self.update_3d_visualization)
        self.solver_thread.finished.connect(self.on_solver_finished)
        self.solver_thread.error.connect(self.on_solver_error)
        self.solver_thread.start()
    
    def on_solver_finished(self, results):
        """Handle completed solver run"""
        self.xyz_iters = results['XYZ_iters']
        self.xyz_final = results['XYZ_final']
        
        msg = f"Solver finished: {len(self.xyz_iters)} iterations\n"
        if results['final_error']:
            msg += f"Final error: {results['final_error']:.6e}"
        
        self.solver_info.setText(msg)
        self.statusBar().showMessage("RMDU solver complete")
        QMessageBox.information(self, "Success", "RMDU solver completed successfully")
        
        # Enable animation if iterations available
        if self.xyz_iters is not None and len(self.xyz_iters) > 1:
            self.frame_label.setText(f"Frame: 0 / {len(self.xyz_iters)-1}")
    
    def on_solver_error(self, error_msg):
        """Handle solver error"""
        self.statusBar().showMessage("Error in solver")
        QMessageBox.critical(self, "Solver Error", error_msg)
        print(f"Solver Error: {error_msg}")
    
    def update_3d_visualization(self, xyz):
        """Update 3D visualization with current iteration (thread-safe via signal)
        
        IMPORTANT: This slot is called from SolverThread via pyqtSignal emission.
        Qt automatically marshals signal emissions to the main thread, ensuring
        all Qt operations here are thread-safe and won't cause timer/parent errors.
        """
        try:
            self.view_3d.clear()
            
            # Add grid and axis
            grid = gl.GLGridItem()
            self.view_3d.addItem(grid)
            axis = gl.GLAxisItem()
            axis.setSize(1, 1, 1)
            self.view_3d.addItem(axis)
            
            # Plot points
            if xyz.shape[0] > 0 and xyz.shape[1] == 3:
                # Use TOA matrix shape to determine number of sources (Ns)
                Ns = self.toa_matrix.shape[0] if self.toa_matrix is not None else 0
                
                if Ns > 0 and xyz.shape[0] > Ns:
                    # Plot Sources (first Ns points) - Green, larger
                    sources_xyz = xyz[:Ns]
                    scatter_sources = gl.GLScatterPlotItem(pos=sources_xyz, color=(0, 1, 0, 1), size=10)
                    self.view_3d.addItem(scatter_sources)
                    
                    # Plot Microphones (remaining points) - Red, standard size
                    mics_xyz = xyz[Ns:]
                    scatter_mics = gl.GLScatterPlotItem(pos=mics_xyz, color=(1, 0, 0, 1), size=5)
                    self.view_3d.addItem(scatter_mics)
                else:
                    # Fallback if dimensions unclear
                    scatter = gl.GLScatterPlotItem(pos=xyz, color=(1, 0, 0, 1), size=5)
                    self.view_3d.addItem(scatter)
        except Exception as e:
            # Log error but don't propagate - visualization is non-critical
            print(f"Warning: Error updating 3D visualization: {e}")
    
    # ==================== TOA Matrix Save/Load ====================
    def save_toa_matrix(self):
        """Save the validated TOA matrix to file"""
        if self.toa_matrix is None:
            QMessageBox.warning(self, "Error", "No validated TOA matrix available. Complete Step 1 first.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Validated TOA Matrix",
            "validated_toa_matrix.npz",
            "NumPy Compressed (*.npz);;NumPy Array (*.npy);;CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                ext = os.path.splitext(file_path)[1].lower()
                
                # Prepare metadata
                metadata = {
                    'timestamp': datetime.now().isoformat(),
                    'temperature_celsius': self.temperature,
                    'toa_matrix_shape': self.toa_matrix.shape,
                    'num_sources': self.toa_matrix.shape[0],
                    'num_microphones': self.toa_matrix.shape[1],
                    'selected_filenames': self.selected_toa_filenames if hasattr(self, 'selected_toa_filenames') else [],
                    'dtoa_matrix': self.dtoa_matrix.tolist() if (hasattr(self, 'dtoa_matrix') and self.dtoa_matrix is not None) else None
                }
                
                if ext == '.npz':
                    # Save as compressed numpy with metadata
                    np.savez_compressed(
                        file_path,
                        toa_matrix=self.toa_matrix,
                        metadata=metadata
                    )
                elif ext == '.npy':
                    # Save just the matrix
                    np.save(file_path, self.toa_matrix)
                else:
                    # Save as CSV
                    np.savetxt(file_path, self.toa_matrix, delimiter=',', fmt='%.6e')
                
                self.statusBar().showMessage(f"TOA matrix saved to {file_path}")
                QMessageBox.information(self, "Success", f"TOA matrix saved successfully.\n\nShape: {self.toa_matrix.shape}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save TOA matrix:\n{str(e)}")
    
    def load_toa_matrix(self):
        """Load a previously saved TOA matrix"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Validated TOA Matrix",
            "",
            "NumPy Compressed (*.npz);;NumPy Array (*.npy);;CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                ext = os.path.splitext(file_path)[1].lower()
                
                if ext == '.npz':
                    # Load from compressed numpy
                    data = np.load(file_path, allow_pickle=True)
                    self.toa_matrix = data['toa_matrix']
                    # Try to load metadata
                    if 'metadata' in data:
                        metadata = data['metadata'].item()
                        if isinstance(metadata.get('selected_filenames'), list):
                            self.selected_toa_filenames = metadata.get('selected_filenames', [])
                        if metadata.get('temperature_celsius'):
                            self.temperature = metadata['temperature_celsius']
                            self.temp_spin.setValue(self.temperature)
                    
                elif ext == '.npy':
                    # Load numpy array
                    self.toa_matrix = np.load(file_path)
                else:
                    # Load CSV
                    self.toa_matrix = np.loadtxt(file_path, delimiter=',')
                
                # Update status
                msg = f"TOA matrix loaded: {self.toa_matrix.shape[0]} sources × {self.toa_matrix.shape[1]} mics"
                self.toa_status.setText(msg)
                self.statusBar().showMessage(msg)
                QMessageBox.information(self, "Success", f"TOA matrix loaded successfully.\n\n{msg}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load TOA matrix:\n{str(e)}")
    
    def load_toa_matrix_for_step2(self):
        """Load a previously saved TOA matrix for use in Step 2"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load TOA Matrix for Geometry Inference",
            "",
            "NumPy Compressed (*.npz);;NumPy Array (*.npy);;CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                ext = os.path.splitext(file_path)[1].lower()
                
                if ext == '.npz':
                    # Load from compressed numpy
                    data = np.load(file_path, allow_pickle=True)
                    self.toa_matrix = data['toa_matrix']
                    # Try to load metadata
                    if 'metadata' in data:
                        metadata = data['metadata'].item()
                        if isinstance(metadata.get('selected_filenames'), list):
                            self.selected_toa_filenames = metadata.get('selected_filenames', [])
                        if metadata.get('temperature_celsius'):
                            self.temperature = metadata['temperature_celsius']
                            self.temp_spin.setValue(self.temperature)
                    
                elif ext == '.npy':
                    # Load numpy array
                    self.toa_matrix = np.load(file_path)
                else:
                    # Load CSV
                    self.toa_matrix = np.loadtxt(file_path, delimiter=',')
                
                # Update status in Step 2
                msg = f"Loaded: {self.toa_matrix.shape[0]} sources, {self.toa_matrix.shape[1]} mics"
                self.loaded_toa_label.setText(msg)
                self.statusBar().showMessage(f"TOA matrix loaded for Step 2: {msg}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load TOA matrix:\n{str(e)}")
    
    # ==================== Geometry Save/Load ====================
    def save_geometry(self):
        """Save the inferred geometry (XYZ final) to file"""
        if self.xyz_final is None:
            QMessageBox.warning(self, "Error", "No inferred geometry available. Run solver first.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Inferred Geometry",
            "geometry_xyz_final.npz",
            "NumPy Compressed (*.npz);;NumPy Array (*.npy);;CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                ext = os.path.splitext(file_path)[1].lower()
                
                # Prepare metadata
                metadata = {
                    'timestamp': datetime.now().isoformat(),
                    'geometry_shape': self.xyz_final.shape,
                    'num_points': self.xyz_final.shape[0],
                    'num_dimensions': self.xyz_final.shape[1] if len(self.xyz_final.shape) > 1 else 1,
                    'temperature_celsius': self.temperature,
                    'lambda_param': self.lambda_param,
                    'num_iterations': len(self.xyz_iters) if self.xyz_iters is not None else 0,
                    'final_error': self.xyz_iters[-1] if self.xyz_iters is not None else None,
                }
                
                if ext == '.npz':
                    # Save as compressed numpy with metadata
                    np.savez_compressed(
                        file_path,
                        xyz_final=self.xyz_final,
                        metadata=metadata
                    )
                elif ext == '.npy':
                    # Save just the matrix
                    np.save(file_path, self.xyz_final)
                else:
                    # Save as CSV
                    np.savetxt(file_path, self.xyz_final, delimiter=',', fmt='%.6e')
                
                self.statusBar().showMessage(f"Geometry saved to {file_path}")
                QMessageBox.information(self, "Success", 
                    f"Geometry saved successfully.\n\nShape: {self.xyz_final.shape}\n"
                    f"Points: {self.xyz_final.shape[0]}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save geometry:\n{str(e)}")
    
    def load_geometry_as_xinit(self):
        """Load a previously saved geometry as initial input (Xinit) for solver"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Geometry as Initial Input (Xinit)",
            "",
            "NumPy Array (*.npy);;NumPy Compressed (*.npz);;CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                ext = os.path.splitext(file_path)[1].lower()
                
                if ext == '.npz':
                    # Load from compressed numpy
                    data = np.load(file_path, allow_pickle=True)
                    self.xyz_init = data['xyz_final']
                    # Try to load metadata
                    if 'metadata' in data:
                        metadata = data['metadata'].item()
                        if metadata.get('temperature_celsius'):
                            temp_info = f" (saved at {metadata['temperature_celsius']}°C)"
                        else:
                            temp_info = ""
                
                elif ext == '.npy':
                    # Load numpy array
                    self.xyz_init = np.load(file_path)
                    temp_info = ""
                    
                else:
                    # Load CSV
                    self.xyz_init = np.loadtxt(file_path, delimiter=',')
                    if self.xyz_init.ndim == 1:
                        self.xyz_init = self.xyz_init.reshape(-1, 1)
                    temp_info = ""
                
                # Update status
                msg = f"Loaded Xinit: {self.xyz_init.shape[0]} points, {self.xyz_init.shape[1] if len(self.xyz_init.shape) > 1 else 1} dims"
                self.loaded_xinit_label.setText(msg + temp_info)
                self.statusBar().showMessage(f"Geometry loaded as Xinit: {msg}")
                QMessageBox.information(self, "Success", 
                    f"Geometry loaded as initial input.\n\n{msg}\n\n"
                    f"Next solver run will use this as starting point.")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load geometry:\n{str(e)}")
    
    # ==================== Alignment ====================
    def enter_reference_positions(self):
        """Enter reference microphone positions"""
        num_ref = self.num_ref_spin.value()
        dialog = AlignmentDialog(self, num_ref_mics=num_ref)
        if dialog.exec_() == QDialog.Accepted:
            self.ref_positions = dialog.get_positions()
            self.align_status.setText(f"Ref positions entered: {num_ref} mics")
    
    def select_microphone_indices(self):
        """Select microphone indices corresponding to reference positions"""
        if self.toa_matrix is None:
            QMessageBox.warning(self, "Error", "No TOA matrix available")
            return
        
        num_ref = self.num_ref_spin.value()
        total_mics = self.toa_matrix.shape[1]
        
        dialog = MicrophonePositionDialog(self, total_mics, num_ref)
        if dialog.exec_() == QDialog.Accepted:
            self.mic_indices = dialog.get_indices()
            self.align_status.setText(f"Mic indices selected: {self.mic_indices}")
    
    def perform_alignment(self):
        """Perform alignment using reference microphone positions"""
        if self.xyz_final is None:
            QMessageBox.warning(self, "Error", "No geometry data available")
            return
        
        if self.ref_positions is None or self.mic_indices is None:
            QMessageBox.warning(self, "Error", "Please enter reference positions and select microphone indices")
            return
        
        try:
            # Extract measured microphone positions
            measured_mics = self.xyz_final[self.mic_indices, :]
            
            # Center both point sets
            ref_centered = self.ref_positions - self.ref_positions.mean(axis=0)
            measured_centered = measured_mics - measured_mics.mean(axis=0)
            
            # Compute rotation matrix using Procrustes
            rotation, rotated = procrustes(measured_centered, ref_centered)
            
            self.alignment_rotation = rotation
            self.alignment_translation = self.ref_positions.mean(axis=0) - measured_mics.mean(axis=0)
            
            # Apply alignment to all geometries
            self.xyz_final = np.dot(self.xyz_final - self.xyz_final.mean(axis=0), rotation) + self.ref_positions.mean(axis=0)
            
            self.align_status.setText("Alignment successful")
            self.statusBar().showMessage("Antenna alignment completed")
            QMessageBox.information(self, "Success", "Antenna alignment completed")
            
            # Update visualization
            self.plot_final_geometry()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Alignment failed: {str(e)}")
    
    # ==================== Visualization ====================
    def plot_toa_matrix(self):
        """Plot the TOA matrix"""
        if self.toa_matrix is None:
            QMessageBox.warning(self, "Error", "No TOA matrix available")
            return
        
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(12, 6))
        plt.pcolormesh(self.toa_matrix.T, cmap='jet')
        plt.colorbar(label='Time of Arrival (s)')
        plt.xlabel('File Index')
        plt.ylabel('Microphone Index')
        plt.title('TOA Matrix')
        plt.tight_layout()
        plt.show()
    
    def plot_final_geometry(self):
        """Plot final geometry in 3D with colors by microphone index"""
        if self.xyz_final is None:
            QMessageBox.warning(self, "Error", "No geometry data available")
            return
        
        self.view_3d.clear()
        
        # Add grid and axis
        grid = gl.GLGridItem()
        self.view_3d.addItem(grid)
        axis = gl.GLAxisItem()
        axis.setSize(1, 1, 1)
        self.view_3d.addItem(axis)
        
        # Determine Ns (Sources) and split if possible
        Ns = getattr(self, 'solver_Ns', 0)
        if Ns == 0 and self.toa_matrix is not None:
             # If toa_matrix is (NbFiles, NbMics), then Ns is NbFiles
             if self.toa_matrix.ndim == 2:
                 Ns = self.toa_matrix.shape[0]
             
        xyz = self.xyz_final
        
        # Sources part
        sources = np.array([])
        mics = xyz
        
        if Ns > 0 and xyz.shape[0] > Ns:
             sources = xyz[:Ns]
             mics = xyz[Ns:]
             
             # Plot Sources (Green, Large) - Using ScatterPlot for stability
             scatter_sources = gl.GLScatterPlotItem(
                 pos=sources, 
                 color=(0, 1, 0, 1), 
                 size=15, 
                 pxMode=True
             )
             self.view_3d.addItem(scatter_sources)
        
        # Plot Microphones (Color by index)
        # Create colors based on microphone index
        num_mics = mics.shape[0]
        color_values = np.arange(num_mics) / (num_mics - 1) if num_mics > 1 else np.array([0.5])
        colors_rgba = plt.cm.hsv(color_values)  # (N, 4) RGBA array
        
        # Use ScatterPlotItem with per-point colors
        scatter_mics = gl.GLScatterPlotItem(
            pos=mics,
            color=colors_rgba,
            size=8,
            pxMode=True
        )
        self.view_3d.addItem(scatter_mics)
        
        # Highlight reference mics if available
        if self.mic_indices is not None:
            ref_mics_pos = []
            ref_colors = []
            for mic_idx in self.mic_indices:
                if mic_idx < len(mics):
                    ref_mics_pos.append(mics[mic_idx])
                    ref_colors.append(colors_rgba[mic_idx])
            
            if ref_mics_pos:
                scatter_refs = gl.GLScatterPlotItem(
                    pos=np.array(ref_mics_pos),
                    color=np.array(ref_colors),
                    size=15,  # Larger
                    pxMode=True
                )
                self.view_3d.addItem(scatter_refs)
    
    # ==================== Animation ====================
    def play_animation(self):
        """Play iteration animation"""
        if self.xyz_iters is None or len(self.xyz_iters) < 2:
            QMessageBox.warning(self, "Error", "Not enough iterations for animation")
            return
        
        self.current_frame = 0
        
        if self.animation_timer is None:
            self.animation_timer = QTimer()
            self.animation_timer.timeout.connect(self.animate_frame)
        
        interval = max(10, 500 - self.speed_slider.value())
        self.animation_timer.start(interval)
        self.statusBar().showMessage("Animation playing...")
    
    def pause_animation(self):
        """Pause animation"""
        if self.animation_timer is not None:
            self.animation_timer.stop()
        self.statusBar().showMessage("Animation paused")
    
    def animate_frame(self):
        """Animate next frame"""
        if self.xyz_iters is None:
            return
        
        max_frames = len(self.xyz_iters)
        
        # Update 3D visualization
        self.view_3d.clear()
        grid = gl.GLGridItem()
        self.view_3d.addItem(grid)
        axis = gl.GLAxisItem()
        axis.setSize(1, 1, 1)
        self.view_3d.addItem(axis)
        
        xyz = self.xyz_iters[self.current_frame]
        
        # Determine Ns (sources) count
        Ns = 0
        if hasattr(self, 'solver_Ns') and self.solver_Ns is not None:
            Ns = self.solver_Ns
        elif self.toa_matrix is not None:
            Ns = self.toa_matrix.shape[0]
            
        if Ns > 0 and xyz.shape[0] > Ns:
            # Plot Sources (first Ns points) - Green, larger
            scatter_sources = gl.GLScatterPlotItem(
                pos=xyz[:Ns],
                color=(0, 1, 0, 1),
                size=10
            )
            self.view_3d.addItem(scatter_sources)
            
            # Plot Microphones (remaining points) - Red, standard size
            scatter_mics = gl.GLScatterPlotItem(
                pos=xyz[Ns:],
                color=(1, 0, 0, 1),
                size=5
            )
            self.view_3d.addItem(scatter_mics)
        else:
            scatter = gl.GLScatterPlotItem(
                pos=xyz,
                color=(1, 0, 0, 1),
                size=8
            )
            self.view_3d.addItem(scatter)
        
        self.frame_label.setText(f"Frame: {self.current_frame} / {max_frames-1}")
        
        self.current_frame += 1
        if self.current_frame >= max_frames:
            self.animation_timer.stop()
            self.statusBar().showMessage("Animation finished")
    
    def save_animation(self):
        """Save animation to video file"""
        if self.xyz_iters is None:
            QMessageBox.warning(self, "Error", "No animation data available")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Animation",
            "",
            "MP4 Files (*.mp4);;AVI Files (*.avi)"
        )
        
        if file_path:
            try:
                self.statusBar().showMessage("Saving animation...")
                from matplotlib.animation import FuncAnimation, FFMpegWriter
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D
                
                L = np.max(np.abs(self.xyz_iters))
                
                fig = plt.figure(facecolor='black')
                ax = fig.add_subplot(111, projection='3d', facecolor='black')
                ax.set_xlim([-L, L])
                ax.set_ylim([-L, L])
                ax.set_zlim([-L, L])
                ax.set_xlabel('X', color='white')
                ax.set_ylabel('Y', color='white')
                ax.set_zlabel('Z', color='white')
                ax.tick_params(colors='white')
                
                scatter = ax.scatter([], [], [], c='r', s=20)
                
                def update(frame):
                    scatter._offsets3d = (
                        self.xyz_iters[frame, :, 0],
                        self.xyz_iters[frame, :, 1],
                        self.xyz_iters[frame, :, 2]
                    )
                    return scatter,
                
                ani = FuncAnimation(fig, update, frames=len(self.xyz_iters), interval=50, blit=False)
                
                ext = os.path.splitext(file_path)[1].lower()
                if ext == '.mp4':
                    writer = FFMpegWriter(fps=20, metadata=dict(artist='GeoCalib'), bitrate=1800)
                else:
                    writer = FFMpegWriter(fps=20, metadata=dict(artist='GeoCalib'), bitrate=1800)
                
                ani.save(file_path, writer=writer)
                plt.close(fig)
                
                self.statusBar().showMessage(f"Animation saved to {file_path}")
                QMessageBox.information(self, "Success", f"Animation saved to:\n{file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save animation:\n{str(e)}")
    
    # ==================== Data Export ====================
    def save_xyz_results(self):
        """Save XYZ results to .npy file"""
        if self.xyz_final is None:
            QMessageBox.warning(self, "Error", "No geometry data to save")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save XYZ Results",
            "XYZ_final.npy",
            "NumPy Files (*.npy);;CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                ext = os.path.splitext(file_path)[1].lower()
                if ext == '.npy':
                    np.save(file_path, self.xyz_final)
                else:
                    np.savetxt(file_path, self.xyz_final, delimiter=',', fmt='%.6f')
                
                self.statusBar().showMessage(f"Saved to {file_path}")
                QMessageBox.information(self, "Success", f"Data saved to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Save failed:\n{str(e)}")
    
    def save_configuration(self):
        """Save processing configuration to JSON"""
        config = {
            'timestamp': datetime.now().isoformat(),
            'temperature_celsius': self.temperature,
            'toa_files_count': len(self.toa_files),
            'toa_matrix_shape': self.toa_matrix.shape if self.toa_matrix is not None else None,
            'lambda_param': self.lambda_param,
            'max_iterations': self.max_iter_spin.value(),
            'eps_limit': self.eps_spin.value(),
            'alignment_performed': self.alignment_rotation is not None,
            'ref_mics_count': len(self.mic_indices) if self.mic_indices is not None else 0
        }
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Configuration",
            "config.json",
            "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                self.statusBar().showMessage(f"Config saved to {file_path}")
                QMessageBox.information(self, "Success", f"Configuration saved to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Save failed:\n{str(e)}")


def main():
    app = QApplication(sys.argv)
    window = GeoCalibApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
