"""
Simple Signal Visualizer with Metadata
========================================

Load and visualize signals from HDF5 or DAT files with metadata display.
"""

import matplotlib.pyplot as plt
import h5py as h5
import numpy as np
import os
import glob
from pathlib import Path
from DATParser import DATParser, LogParser


def get_data_files(directory=None):
    """Find all HDF5 and DAT files in directory"""
    if directory is None:
        # Look in current and parent directories
        for search_dir in ['.', '..', '../../DataCalib4', './data']:
            if os.path.isdir(search_dir):
                h5_files = sorted(glob.glob(os.path.join(search_dir, '*.h5')))
                dat_files = sorted(glob.glob(os.path.join(search_dir, '*.dat')))
                if h5_files or dat_files:
                    return h5_files + dat_files
    else:
        h5_files = sorted(glob.glob(os.path.join(directory, '*.h5')))
        dat_files = sorted(glob.glob(os.path.join(directory, '*.dat')))
        return h5_files + dat_files
    
    return []


def print_hdf5_metadata(file_path):
    """Extract and print metadata from HDF5 file"""
    print("\n" + "="*60)
    print(f"FILE: {os.path.basename(file_path)}")
    print("="*60)
    
    try:
        with h5.File(file_path, 'r') as data:
            print(f"HDF5 File Structure:")
            print(f"  Root keys: {list(data.keys())}")
            
            if 'muh5' in data:
                sections = sorted([int(i) for i in data['muh5'].keys()])
                print(f"  Number of sections: {len(sections)}")
                
                # Get shape from first section
                first_key = str(sections[0])
                if 'sig' in data['muh5'][first_key]:
                    sig_shape = data['muh5'][first_key]['sig'].shape
                    print(f"  First section shape: {sig_shape}")
                    print(f"  Channels: {sig_shape[0]}, Samples: {sig_shape[1]}")
                    
                    # Calculate total samples
                    total_samples = sum(data['muh5'][str(i)]['sig'].shape[1] for i in sections)
                    print(f"  Total samples (all sections): {total_samples}")
            
            # Print other attributes
            for attr_name in data.attrs:
                print(f"  Attr '{attr_name}': {data.attrs[attr_name]}")
    
    except Exception as e:
        print(f"  Error reading file: {e}")


def print_dat_metadata(file_path):
    """Extract and print metadata from DAT file and associated LOG"""
    print("\n" + "="*60)
    print(f"FILE: {os.path.basename(file_path)}")
    print("="*60)
    
    log_file = file_path.replace('.dat', '.log')
    if not os.path.exists(log_file):
        print(f"  ERROR: Log file not found: {log_file}")
        return
    
    try:
        parser = LogParser(log_file)
        metadata = parser.metadata
        
        print(f"LOG Metadata:")
        print(f"  Date: {metadata.get('date', 'N/A')}")
        print(f"  Time: {metadata.get('time', 'N/A')}")
        print(f"  Sampling frequency: {metadata.get('freq', 'N/A')} Hz")
        print(f"  Duration: {metadata.get('duree', 'N/A')} s")
        print(f"  Number of samples: {metadata.get('nb_ech', 'N/A')}")
        print(f"  Type: {metadata.get('type', 'N/A')}")
        print(f"  Active microphones: {metadata.get('nb_micros_actifs', 'N/A')}")
        print(f"  Active analog channels: {metadata.get('nb_voies_analogiques', 'N/A')}")
        
        # DAT file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"  File size: {file_size_mb:.2f} MB")
        
    except Exception as e:
        print(f"  Error reading metadata: {e}")


def visualize_hdf5_file(file_path, max_channels=16):
    """Visualize signals from HDF5 file"""
    print(f"\nVisualizing {os.path.basename(file_path)}...")
    
    try:
        with h5.File(file_path, 'r') as data:
            if 'muh5' not in data:
                print("  No 'muh5' dataset found")
                return
            
            sections = sorted([int(i) for i in data['muh5'].keys()])
            
            # Load all sections and concatenate
            all_data = []
            for section_idx in sections:
                section_data = data['muh5'][str(section_idx)]['sig'][:]
                all_data.append(section_data)
            
            # Concatenate along time axis
            complete_signal = np.concatenate(all_data, axis=1)
            total_channels = complete_signal.shape[0]
            total_samples = complete_signal.shape[1]
            
            print(f"  Loaded {len(sections)} sections: {total_samples} total samples")
            
            # Extract channels: index 0 is counter, 1-256 are mics, 257+ are analog
            mics = complete_signal[1:257, :]  # Microphones (256 channels)
            analog = complete_signal[257:, :]  # Analog channels from 257 onwards
            
            # Combine: analog first, then mics
            if analog.shape[0] > 0:
                all_signals = np.vstack([analog, mics])
                num_analog = analog.shape[0]
            else:
                all_signals = mics
                num_analog = 0
            
            channels_to_plot = min(max_channels, all_signals.shape[0])
            
            # Create figure
            fig, axes = plt.subplots(channels_to_plot, 1, figsize=(12, 2*channels_to_plot))
            if channels_to_plot == 1:
                axes = [axes]
            
            # Plot each channel
            for ch in range(channels_to_plot):
                signal = all_signals[ch, :]
                if ch < num_analog:
                    label = f'Analog {ch} (ch {257 + ch})'
                else:
                    label = f'Mic {ch - num_analog} (ch {1 + ch - num_analog})'
                
                axes[ch].plot(signal, linewidth=0.5)
                axes[ch].set_ylabel(label)
                axes[ch].grid(True, alpha=0.3)
                
                # Add statistics
                min_val, max_val = np.min(signal), np.max(signal)
                axes[ch].text(0.02, 0.95, f'min={min_val:.0f}, max={max_val:.0f}',
                            transform=axes[ch].transAxes, fontsize=8,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            axes[-1].set_xlabel('Sample')
            fig.suptitle(f'{os.path.basename(file_path)} - Complete Duration ({total_samples} samples, {num_analog} analog + {mics.shape[0]} mics)', 
                         fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.show()
    
    except Exception as e:
        print(f"  Error visualizing file: {e}")


def visualize_dat_file(file_path, max_channels=16):
    """Visualize signals from DAT file"""
    print(f"\nVisualizing {os.path.basename(file_path)}...")
    
    log_file = file_path.replace('.dat', '.log')
    if not os.path.exists(log_file):
        print(f"  ERROR: Log file not found: {log_file}")
        return
    
    try:
        parser = DATParser(file_path, log_file)
        mics = parser.extract_microphone_signals()
        vas = parser.extract_analog_signals()
        
        # Combine signals: analog channels first, then microphone signals
        if vas.size > 0:
            all_signals = np.vstack([vas.T, mics])
            num_analog = vas.shape[1]
        else:
            all_signals = mics
            num_analog = 0
        
        channels_available = all_signals.shape[0]
        channels_to_plot = min(max_channels, channels_available)
        
        # Create figure
        fig, axes = plt.subplots(channels_to_plot, 1, figsize=(12, 2*channels_to_plot))
        if channels_to_plot == 1:
            axes = [axes]
        
        # Plot each channel
        for ch in range(channels_to_plot):
            signal = all_signals[ch, :]
            if ch < num_analog:
                label = f'Analog {ch}'
            else:
                label = f'Mic {ch - num_analog}'
            
            axes[ch].plot(signal, linewidth=0.5)
            axes[ch].set_ylabel(label)
            axes[ch].grid(True, alpha=0.3)
            
            # Add statistics
            min_val, max_val = np.min(signal), np.max(signal)
            axes[ch].text(0.02, 0.95, f'min={min_val:.0f}, max={max_val:.0f}',
                        transform=axes[ch].transAxes, fontsize=8,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        axes[-1].set_xlabel('Sample')
        fig.suptitle(f'{os.path.basename(file_path)} ({num_analog} analog + {mics.shape[0]} mics)', 
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    except Exception as e:
        print(f"  Error visualizing file: {e}")


def main():
    """Main visualization loop"""
    # Find data files
    files = get_data_files()
    
    if not files:
        print("No data files found. Checking current directory...")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files: {os.listdir('.')}")
        return
    
    print(f"\nFound {len(files)} data files:")
    for i, file_path in enumerate(files):
        print(f"  {i+1}. {os.path.basename(file_path)}")
    
    # Process each file
    for file_path in files:
        try:
            if file_path.lower().endswith(('.h5', '.hdf5')):
                print_hdf5_metadata(file_path)
                visualize_hdf5_file(file_path, max_channels=8)
            
            elif file_path.lower().endswith('.dat'):
                print_dat_metadata(file_path)
                visualize_dat_file(file_path, max_channels=8)
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print("\n" + "="*60)
    print("Visualization complete")
    print("="*60)


if __name__ == '__main__':
    main()
