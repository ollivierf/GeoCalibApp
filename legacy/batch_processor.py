#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch processing script for automated geocalibration
Can be used in headless mode or integrated with existing workflows
"""

import os
import numpy as np
import argparse
import json
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from GeoCalibUtils import procrustes, evalDTOA, Iters2PltAnimation

try:
    import rcbox
    from rcbox.rmds import RMDU
    RCBOX_AVAILABLE = True
except ImportError:
    RCBOX_AVAILABLE = False
    print("Warning: rcbox not available. Install with: pip install rcbox")

import h5py
import scipy.signal as sig
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class BatchGeoCalibProcessor:
    """Batch processor for geocalibration"""
    
    def __init__(self, config_file=None):
        """Initialize processor with optional config file"""
        self.config = self._load_default_config()
        if config_file and os.path.exists(config_file):
            self._load_config(config_file)
        
        self.toa_matrix = None
        self.xyz_iters = None
        self.xyz_final = None
        self.toa_files = []
        
    def _load_default_config(self):
        """Load default configuration"""
        return {
            'temperature_celsius': 26.0,
            'max_distance_m': 3.0,
            'lambda_param': 0.050,
            'max_iterations': 10000,
            'eps_limit': 1e-10,
            'verbose': True
        }
    
    def _load_config(self, config_file):
        """Load configuration from JSON file"""
        with open(config_file, 'r') as f:
            config = json.load(f)
        self.config.update(config)
    
    def process_toa_files(self, file_list):
        """Process TOA from calibration files"""
        print(f"Processing {len(file_list)} files...")
        
        Tc = self.config['temperature_celsius']
        C = np.sqrt(1.4 * 287 * (Tc + 273))
        Fe = 50e3
        tmax = self.config['max_distance_m'] / C
        
        toa_list = []
        
        for i, file_path in enumerate(file_list):
            print(f"  [{i+1}/{len(file_list)}] {os.path.basename(file_path)}", end=' ')
            
            try:
                data = h5py.File(file_path, 'r')
                
                Secs = [int(j) for j in data['muh5'].keys()]
                Sig = np.concatenate([data['muh5'][str(j)]['sig'][:] for j in range(len(Secs))], axis=1)
                
                Cmptr = Sig[0, :]
                if np.sum(np.diff(Cmptr)) != len(Cmptr) - 1:
                    print("SKIPPED (bad counter)")
                    data.close()
                    continue
                
                Mics = Sig[np.arange(1, 257), :]
                Mics = np.vstack((Mics, Sig[259, :]))
                Ref = -Sig[257, :]
                
                NbTixels = len(Cmptr)
                NFFT = NbTixels
                df = Fe / NFFT
                
                SRef = np.fft.rfft(Ref, NFFT)
                SMics = np.fft.rfft(Mics, NFFT)
                GCS = np.conj(SRef)[None, :] * (SMics) / (np.abs(SMics) * np.abs(SRef)[None, :])
                
                NUp = 10 * NFFT
                dtUp = 1 / (NUp * df)
                GCC = np.fft.irfft(GCS, NUp)
                tt = np.arange(NUp) * dtUp
                
                GCCE = np.abs(sig.hilbert(GCC[:, tt < tmax], axis=1))
                tt = tt[tt < tmax]
                
                imax = np.argmax(GCCE[:, 500:-100], axis=1) + 500
                toas = np.array([tt[i] for i in imax])
                
                toa_list.append(toas)
                print("OK")
                data.close()
                
            except Exception as e:
                print(f"FAILED ({str(e)})")
                continue
        
        if len(toa_list) == 0:
            raise ValueError("No valid TOA data processed")
        
        self.toa_matrix = np.array(toa_list)
        self.toa_files = file_list
        print(f"TOA matrix shape: {self.toa_matrix.shape}")
        return self.toa_matrix
    
    def run_solver(self, xyz_init=None):
        """
        Run RMDU solver on TOA matrix
        
        Parameters
        ----------
        xyz_init : ndarray, optional
            Initial microphone positions
        """
        if self.toa_matrix is None:
            raise ValueError("No TOA matrix. Process files first.")
        
        if not RCBOX_AVAILABLE:
            raise RuntimeError("rcbox package required. Install with: pip install rcbox")
        
        print("\nRunning RMDU solver...")
        
        Tc = self.config['temperature_celsius']
        C = np.sqrt(1.4 * 287 * (Tc + 273))
        
        distance_matrix = C * self.toa_matrix.T
        
        solver = RMDU(distance_matrix)
        solver.Run(
            lbda=self.config['lambda_param'],
            Xinit=xyz_init,
            itmax=self.config['max_iterations'],
            EpsLim=self.config['eps_limit'],
            verbose=1 if self.config['verbose'] else 0
        )
        
        self.xyz_iters = solver.X.copy()
        self.xyz_final = self.xyz_iters[-1, ...].copy()
        
        print(f"Solver completed: {len(self.xyz_iters)} iterations")
        return self.xyz_final
    
    def align_geometry(self, ref_positions, mic_indices):
        """
        Align geometry to reference positions
        
        Parameters
        ----------
        ref_positions : array-like, shape (n, 3)
            Reference microphone positions
        mic_indices : array-like, shape (n,)
            Indices of mics in the array corresponding to references
        """
        if self.xyz_final is None:
            raise ValueError("No geometry data. Run solver first.")
        
        print("\nPerforming alignment...")
        
        ref_positions = np.array(ref_positions)
        mic_indices = np.array(mic_indices)
        
        measured_mics = self.xyz_final[mic_indices, :]
        
        ref_centered = ref_positions - ref_positions.mean(axis=0)
        measured_centered = measured_mics - measured_mics.mean(axis=0)
        
        rotation, rotated = procrustes(measured_centered, ref_centered)
        
        # Apply alignment
        xyz_aligned = np.dot(self.xyz_final - self.xyz_final.mean(axis=0), rotation)
        xyz_aligned += ref_positions.mean(axis=0)
        
        self.xyz_final = xyz_aligned
        
        print(f"Alignment complete. Rotation matrix:\n{rotation}")
        return xyz_aligned
    
    def save_results(self, output_dir):
        """Save all results to output directory"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving results to {output_dir}...")
        
        if self.toa_matrix is not None:
            np.save(os.path.join(output_dir, 'toa_matrix.npy'), self.toa_matrix)
            print(f"  Saved: toa_matrix.npy ({self.toa_matrix.shape})")
        
        if self.xyz_final is not None:
            np.save(os.path.join(output_dir, 'xyz_final.npy'), self.xyz_final)
            print(f"  Saved: xyz_final.npy ({self.xyz_final.shape})")
        
        if self.xyz_iters is not None:
            np.save(os.path.join(output_dir, 'xyz_iterations.npy'), self.xyz_iters)
            print(f"  Saved: xyz_iterations.npy ({self.xyz_iters.shape})")
        
        # Save configuration
        config_path = os.path.join(output_dir, 'processing_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"  Saved: processing_config.json")
        
        # Save file list
        files_path = os.path.join(output_dir, 'processed_files.txt')
        with open(files_path, 'w') as f:
            for file in self.toa_files:
                f.write(f"{file}\n")
        print(f"  Saved: processed_files.txt ({len(self.toa_files)} files)")
    
    def generate_animation(self, output_path):
        """Generate animation video of iterations"""
        if self.xyz_iters is None:
            print("No iterations to animate")
            return
        
        print(f"\nGenerating animation: {output_path}...")
        
        try:
            Iters2PltAnimation(self.xyz_iters, step=1)
            print(f"Animation saved: {output_path}")
        except Exception as e:
            print(f"Animation generation failed: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description='Batch geocalibration processing'
    )
    parser.add_argument(
        'input_dir',
        help='Directory containing HDF5 calibration files'
    )
    parser.add_argument(
        '-o', '--output',
        default='./geocalib_output',
        help='Output directory (default: ./geocalib_output)'
    )
    parser.add_argument(
        '-c', '--config',
        help='Configuration JSON file'
    )
    parser.add_argument(
        '-t', '--temperature',
        type=float,
        default=26.0,
        help='Temperature in Celsius (default: 26.0)'
    )
    parser.add_argument(
        '-a', '--animate',
        action='store_true',
        help='Generate animation of iterations'
    )
    parser.add_argument(
        '-r', '--ref-positions',
        help='JSON file with reference microphone positions'
    )
    
    args = parser.parse_args()
    
    # Find HDF5 files
    input_path = Path(args.input_dir)
    h5_files = list(input_path.glob('*.h5')) + list(input_path.glob('*.hdf5'))
    
    if not h5_files:
        print(f"Error: No HDF5 files found in {args.input_dir}")
        return 1
    
    print(f"Found {len(h5_files)} HDF5 files")
    
    # Initialize processor
    processor = BatchGeoCalibProcessor(args.config)
    processor.config['temperature_celsius'] = args.temperature
    
    try:
        # Process TOA
        processor.process_toa_files(sorted([str(f) for f in h5_files]))
        
        # Run solver
        processor.run_solver()
        
        # Align if reference data provided
        if args.ref_positions and os.path.exists(args.ref_positions):
            with open(args.ref_positions, 'r') as f:
                ref_data = json.load(f)
            processor.align_geometry(
                ref_data['positions'],
                ref_data['mic_indices']
            )
        
        # Generate animation
        if args.animate:
            anim_path = os.path.join(args.output, 'animation.mp4')
            processor.generate_animation(anim_path)
        
        # Save results
        processor.save_results(args.output)
        
        print("\n✓ Batch processing completed successfully")
        return 0
        
    except Exception as e:
        print(f"\n✗ Processing failed: {str(e)}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
