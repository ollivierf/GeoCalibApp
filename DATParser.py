"""
DAT File Parser for GeoCalib Application
==========================================

Parses .DAT binary files using metadata from corresponding .LOG files.

Format:
    For each sampling instant:
    [Bloc A: counter + mics + VAs + status] + [Bloc B] + [Bloc C] + [Bloc D]
    
    Each bloc contains:
    - Counter (1 x int32)
    - Active microphones (up to 256 x int32)
    - Active analog channels (up to 4 x int32)
    - Status (1 x int32)
"""

import numpy as np
import configparser
from pathlib import Path
from typing import Dict, Tuple, List


class LogParser:
    """Parse .LOG metadata file for DAT structure information."""
    
    def __init__(self, log_file: str):
        """
        Initialize LogParser.
        
        Args:
            log_file: Path to .log file
        """
        self.log_file = Path(log_file)
        self.config = configparser.ConfigParser()
        self.metadata = {}
        self._parse_log()
    
    def _parse_log(self):
        """Parse LOG file and extract metadata."""
        self.config.read(self.log_file, encoding='utf-8-sig')
        
        # Date/Time
        self.metadata['date'] = self.config.get('Date', 'Date', fallback=None)
        self.metadata['time'] = self.config.get('Date', 'Heure', fallback=None)
        
        # Channel Configuration
        self.metadata['nb_faisceaux'] = self.config.getint('Voies', 'Nb_faisceaux')
        self.metadata['nb_voies'] = self.config.getint('Voies', 'Nb_voies')
        self.metadata['nb_micros_actifs'] = self.config.getint('Voies', 'Nb_micros_actifs')
        self.metadata['nb_voies_analogiques'] = self.config.getint('Voies', 'Nb_voies_analogiques')
        self.metadata['nb_compteurs'] = self.config.getint('Voies', 'Nb_compteurs')
        
        # Active beams per bloc
        self.metadata['faisceaux_actifs'] = self._parse_faisceaux()
        
        # Active analog channels per bloc
        self.metadata['va_actives'] = self._parse_va_actives()
        
        # Active counters
        self.metadata['compteurs_actifs'] = self._parse_compteurs()
        
        # Acquisition parameters
        self.metadata['freq'] = self.config.getint('Acquisition', 'Freq')
        self.metadata['duree'] = self.config.getfloat('Acquisition', 'Duree')
        self.metadata['nb_ech'] = self.config.getint('Acquisition', 'Nb_ech')
        self.metadata['type'] = self.config.get('Acquisition', 'Type')
        
    def _parse_faisceaux(self) -> Dict[str, List[int]]:
        """Parse active beam configuration."""
        faisceaux = {}
        for bloc in ['A', 'B', 'C', 'D']:
            try:
                line = self.config.get('Micros actifs', f'Faisceau_{int(bloc, 16):03d}', fallback='')
                if line:
                    faisceaux[bloc] = [int(x) for x in line.split()]
            except:
                pass
        
        # Count active faisceaux for each bloc
        active_counts = {}
        for bloc_idx, bloc in enumerate(['A', 'B', 'C', 'D']):
            count = 0
            for i in range(self.metadata['nb_faisceaux']):
                key = f'Faisceau_{bloc_idx * 8 + i:03d}'
                try:
                    line = self.config.get('Micros actifs', key, fallback='0 0 0 0 0 0 0 0')
                    values = [int(x) for x in line.split()]
                    count += sum(values)
                except:
                    pass
            active_counts[bloc] = count
        
        return active_counts
    
    def _parse_va_actives(self) -> Dict[str, int]:
        """Parse active analog channels per bloc."""
        va_actives = {}
        for bloc in ['A', 'B', 'C', 'D']:
            try:
                line = self.config.get('VA actives', f'Bloc_{bloc}', fallback='0 0 0 0')
                values = [int(x) for x in line.split()]
                va_actives[bloc] = sum(values)
            except:
                va_actives[bloc] = 0
        return va_actives
    
    def _parse_compteurs(self) -> int:
        """Parse active counter configuration."""
        try:
            line = self.config.get('Compteurs actifs', 'Compteurs', fallback='0 0 0 0')
            values = [int(x) for x in line.split()]
            return sum(values)
        except:
            return 1
    
    def get_sample_size(self) -> int:
        """
        Calculate size of one sample in bytes.
        
        Sample = Bloc_A + Bloc_B + Bloc_C + Bloc_D
        Each Bloc = counter + mics + VAs + status
        """
        size_per_bloc = 0
        
        for bloc in ['A', 'B', 'C', 'D']:
            # Counter (1 x int32)
            size_per_bloc += 1
            # Microphones (active count x int32)
            size_per_bloc += self.metadata['faisceaux_actifs'].get(bloc, 0) * 8
            # Analog channels (active count x int32)
            size_per_bloc += self.metadata['va_actives'].get(bloc, 0)
            # Status (1 x int32)
            size_per_bloc += 1
        
        return size_per_bloc * 4  # 4 bytes per int32
    
    def get_bloc_structure(self, bloc: str) -> Dict:
        """Get channel structure for a specific bloc."""
        return {
            'bloc': bloc,
            'counter': 1,
            'mics': self.metadata['faisceaux_actifs'].get(bloc, 0) * 8,
            'vas': self.metadata['va_actives'].get(bloc, 0),
            'status': 1,
            'total_channels': 1 + (self.metadata['faisceaux_actifs'].get(bloc, 0) * 8) + 
                            self.metadata['va_actives'].get(bloc, 0) + 1
        }
    
    def print_metadata(self):
        """Print parsed metadata."""
        print("\n=== DAT File Metadata ===")
        print(f"Date: {self.metadata['date']} {self.metadata['time']}")
        print(f"Sampling Freq: {self.metadata['freq']} Hz")
        print(f"Duration: {self.metadata['duree']} s")
        print(f"Total Samples: {self.metadata['nb_ech']}")
        print(f"Total Channels (per sample): {self.metadata['nb_voies']}")
        print(f"Active Microphones: {self.metadata['nb_micros_actifs']}")
        print(f"Active Analog Channels: {self.metadata['nb_voies_analogiques']}")
        print(f"\nFaisceaux actifs par bloc:")
        for bloc in ['A', 'B', 'C', 'D']:
            mics = self.metadata['faisceaux_actifs'].get(bloc, 0) * 8
            vas = self.metadata['va_actives'].get(bloc, 0)
            print(f"  Bloc {bloc}: {mics} mics, {vas} VAs")
        print(f"\nSample size: {self.get_sample_size()} bytes")


class DATParser:
    """Parse .DAT binary files using LOG metadata."""
    
    def __init__(self, dat_file: str, log_file: str):
        """
        Initialize DATParser.
        
        Args:
            dat_file: Path to .dat binary file
            log_file: Path to corresponding .log metadata file
        """
        self.dat_file = Path(dat_file)
        self.log_parser = LogParser(log_file)
        self.sample_size = self.log_parser.get_sample_size() // 4  # in int32 units
        
    def read_all_samples(self) -> Tuple[np.ndarray, Dict]:
        """
        Read all samples from DAT file.
        
        Returns:
            data: Array of shape (nb_ech, total_channels) with int32 values
            metadata: Dictionary with parsing metadata
        """
        nb_ech = self.log_parser.metadata['nb_ech']
        
        # Read binary file
        with open(self.dat_file, 'rb') as f:
            raw_data = np.frombuffer(f.read(), dtype=np.int32)
        
        # Reshape: each sample has sample_size int32 values
        if len(raw_data) != nb_ech * self.sample_size:
            raise ValueError(
                f"File size mismatch: expected {nb_ech * self.sample_size} values, "
                f"got {len(raw_data)}"
            )
        
        data = raw_data.reshape((nb_ech, self.sample_size))
        
        metadata = {
            'nb_samples': nb_ech,
            'channels_per_sample': self.sample_size,
            'freq': self.log_parser.metadata['freq'],
            'duration': self.log_parser.metadata['duree']
        }
        
        return data, metadata
    
    def extract_microphone_signals(self) -> np.ndarray:
        """
        Extract microphone signals from multiplexed DAT data.
        
        Returns:
            Array of shape (nb_ech, total_active_mics) with microphone signals
        """
        data, _ = self.read_all_samples()
        
        # Extract microphone channels from each bloc
        mic_signals = []
        col_idx = 0
        
        for bloc in ['A', 'B', 'C', 'D']:
            structure = self.log_parser.get_bloc_structure(bloc)
            
            # Skip counter (1 channel)
            col_idx += 1
            
            # Extract mics
            num_mics = structure['mics']
            if num_mics > 0:
                mic_data = data[:, col_idx:col_idx + num_mics]
                mic_signals.append(mic_data)
                col_idx += num_mics
            
            # Skip VAs and status
            col_idx += structure['vas'] + 1
        
        # Concatenate all microphone signals
        if mic_signals:
            return np.concatenate(mic_signals, axis=1)
        else:
            return np.array([])
    
    def extract_analog_signals(self) -> np.ndarray:
        """
        Extract analog channel signals.
        
        Returns:
            Array of shape (nb_ech, total_active_vas) with analog signals
        """
        data, _ = self.read_all_samples()
        
        # Extract analog channels from each bloc
        analog_signals = []
        col_idx = 0
        
        for bloc in ['A', 'B', 'C', 'D']:
            structure = self.log_parser.get_bloc_structure(bloc)
            
            # Skip counter and mics
            col_idx += 1 + structure['mics']
            
            # Extract VAs
            num_vas = structure['vas']
            if num_vas > 0:
                va_data = data[:, col_idx:col_idx + num_vas]
                analog_signals.append(va_data)
                col_idx += num_vas
            
            # Skip status
            col_idx += 1
        
        # Concatenate all analog signals
        if analog_signals:
            return np.concatenate(analog_signals, axis=1)
        else:
            return np.array([])
    
    def extract_counters(self) -> np.ndarray:
        """
        Extract counter/timing signals.
        
        Returns:
            Array of shape (nb_ech,) or (nb_ech, num_counters) with counter values
        """
        data, _ = self.read_all_samples()
        
        # First value of each sample is counter (from Bloc A)
        return data[:, 0]
    
    def get_metadata(self) -> Dict:
        """Get complete metadata."""
        return {
            **self.log_parser.metadata,
            'sample_size': self.sample_size,
            'dat_file': str(self.dat_file),
            'log_file': str(self.log_parser.log_file)
        }


# Example usage
if __name__ == "__main__":
    # Example: Parse Exemple.log and read corresponding .dat
    log_file = "Exemple.log"
    dat_file = "Exemple.dat"  # Must exist
    
    # Parse metadata
    log_parser = LogParser(log_file)
    log_parser.print_metadata()
    
    # Parse DAT file (if it exists)
    try:
        dat_parser = DATParser(dat_file, log_file)
        print(f"\n=== DAT File Parsing ===")
        print(f"File: {dat_file}")
        
        # Read microphone signals
        mics = dat_parser.extract_microphone_signals()
        print(f"Microphone signals shape: {mics.shape}")
        
        # Read analog signals
        vas = dat_parser.extract_analog_signals()
        print(f"Analog signals shape: {vas.shape}")
        
        # Read counters
        counters = dat_parser.extract_counters()
        print(f"Counter values shape: {counters.shape}")
        
    except FileNotFoundError:
        print(f"\nDAT file '{dat_file}' not found. Using log file only.")
