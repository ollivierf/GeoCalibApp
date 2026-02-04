"""
DAT File Parser for GeoCalib Application
==========================================

Rebuilt to strictly follow the LogStructure.txt and .log file definitions.
Structure:
- Samples are sequential.
- Each sample contains `Nb_voies` int32 values.
- Channel order is determined by iterating Blocs [A, B, C, D]:
    1. Counter (if active)
    2. Microphones (active ones, in order of Faisceau 00-31 for A, etc.)
    3. Analog Channels (active ones, in order 1-4)

"""

import numpy as np
import configparser
from pathlib import Path
from typing import Dict, List, Tuple, Any

class LogParser:
    """Parses .LOG metadata file for DAT structure information."""

    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.metadata = {}
        self.mic_matrix = None  # numpy array of shape (128, 8)
        self.bloc_config = {
            'A': {'f_start': 0, 'f_end': 32},
            'B': {'f_start': 32, 'f_end': 64},
            'C': {'f_start': 64, 'f_end': 96},
            'D': {'f_start': 96, 'f_end': 128},
        }
        self.channel_map = [] # List of dicts describing each channel's origin
        
        self._parse_log_file()
        self._build_channel_map()

    def _clean_line(self, value: str) -> str:
        """Remove comments and whitespace."""
        if value is None:
            return ""
        return value.split(';')[0].strip()

    def _parse_int(self, value: str) -> int:
        clean = self._clean_line(value)
        if not clean:
            return 0
        return int(clean)

    def _parse_float(self, value: str) -> float:
        clean = self._clean_line(value)
        if not clean:
            return 0.0
        return float(clean.replace(',', '.'))

    def _parse_log_file(self):
        """Manually parse log file to handle non-standard spacing/keys."""
        current_section = None
        raw_data = {}

        # Read all lines
        with open(self.log_file, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Section detection
            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1]
                # Normalize section names
                if current_section == 'Micros_actifs': current_section = 'Micros actifs'
                # Handle plural/singular variations
                if current_section == 'Compteur actifs': current_section = 'Compteurs actifs'
                continue

            # Key-Value parsing
            if '=' in line:
                key, val = line.split('=', 1)
                self._store_value(current_section, key.strip(), val.strip(), raw_data)
            elif current_section == 'Micros actifs' and line.startswith('Faisceau_'):
                # Handle space-separated matrix lines
                parts = line.split()
                key = parts[0]
                val = " ".join(parts[1:])
                self._store_value(current_section, key, val, raw_data)

        self.raw_data = raw_data
        self._process_metadata(raw_data)

    def _store_value(self, section, key, val, data):
        if section not in data:
            data[section] = {}
        data[section][key] = val

    def _process_metadata(self, data):
        # [Voies]
        voies = data.get('Voies', {})
        self.metadata['nb_faisceaux'] = self._parse_int(voies.get('Nb_faisceaux', '128'))
        self.metadata['nb_voies'] = self._parse_int(voies.get('Nb_voies', '0'))
        self.metadata['nb_micros_actifs'] = self._parse_int(voies.get('Nb_micros_actifs', '0'))
        self.metadata['nb_voies_analogiques'] = self._parse_int(voies.get('Nb_voies_analogiques', '0'))
        self.metadata['nb_compteurs'] = self._parse_int(voies.get('Nb_compteurs', '0'))

        # [Acquisition]
        acq = data.get('Acquisition', {})
        self.metadata['freq'] = self._parse_int(acq.get('Freq', '0'))
        self.metadata['duree'] = self._parse_float(acq.get('Duree', '0.0'))
        self.metadata['nb_ech'] = self._parse_int(acq.get('Nb_ech', '0'))
        self.metadata['type'] = self._clean_line(acq.get('Type', 'int32'))

        # [Compteurs actifs]
        # Search for 'Compteurs' (plural) or 'Compteur' (singular assumed from LogStructure description if misnamed keys exist, but usually keys are 'Compteurs')
        ctrs_sec = data.get('Compteurs actifs', {})
        ctrs_line = ctrs_sec.get('Compteurs', ctrs_sec.get('Compteur', '0 0 0 0'))
        self.metadata['compteurs_map'] = [int(x) for x in self._clean_line(ctrs_line).split()]

        # [VA actives]
        va_sec = data.get('VA actives', {})
        self.metadata['va_map'] = {}
        for bloc in ['A', 'B', 'C', 'D']:
            line = va_sec.get(f'Bloc_{bloc}', '0 0 0 0')
            self.metadata['va_map'][bloc] = [int(x) for x in self._clean_line(line).split()]

        # [Micros actifs]
        self._process_mic_matrix(data.get('Micros actifs', {}))

    def _process_mic_matrix(self, mic_data):
        # Initialize 128 x 8 matrix with zeros
        matrix = np.zeros((128, 8), dtype=int)
        
        for key, val in mic_data.items():
            if key.startswith('Faisceau_'):
                try:
                    idx = int(key.split('_')[1])
                    # Handle both space and tab separation
                    bits = [int(x) for x in self._clean_line(val).split()]
                    # Ensure we take only first 8 digits if more provided
                    if len(bits) >= 8:
                        matrix[idx] = bits[:8]
                except ValueError:
                    pass
        
        self.mic_matrix = matrix
        
        # Validation
        calculated_mics = np.sum(matrix)
        if calculated_mics != self.metadata['nb_micros_actifs']:
            raise ValueError(f"Metadata mismatch: Nb_micros_actifs={self.metadata['nb_micros_actifs']} "
                             f"but found {calculated_mics} active mics in [Micros actifs] matrix.")

    def _build_channel_map(self):
        """
        Construct the order of channels in the DAT file.
        Order per Bloc: [Counter] -> [Mics] -> [VAs]
        """
        self.channel_map = []
        
        bloc_indices = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        
        for bloc in ['A', 'B', 'C', 'D']:
            # 1. Counter
            if len(self.metadata['compteurs_map']) > bloc_indices[bloc]:
                if self.metadata['compteurs_map'][bloc_indices[bloc]] == 1:
                    self.channel_map.append({'type': 'counter', 'bloc': bloc, 'id': 0})

            # 2. Microphones
            f_start = self.bloc_config[bloc]['f_start']
            f_end = self.bloc_config[bloc]['f_end']
            
            for f_idx in range(f_start, f_end):
                for m_idx in range(8):
                    if self.mic_matrix[f_idx, m_idx] == 1:
                        self.channel_map.append({
                            'type': 'mic', 
                            'bloc': bloc, 
                            'faisceau': f_idx, 
                            'mic_index': m_idx
                        })

            # 3. Analog Channels
            va_flags = self.metadata['va_map'].get(bloc, [0,0,0,0])
            for i, active in enumerate(va_flags):
                if active == 1:
                    self.channel_map.append({'type': 'va', 'bloc': bloc, 'id': i})

        # Final validation
        if len(self.channel_map) != self.metadata['nb_voies']:
            # Warn but don't crash, prefer calculated map length
            print(f"Warning: Calculated channel count ({len(self.channel_map)}) "
                  f"differs from Nb_voies ({self.metadata['nb_voies']}). "
                  f"Using calculated map.")
            self.metadata['nb_voies'] = len(self.channel_map)

class DATParser:
    """Read binary DAT file using structure from LogParser."""
    
    def __init__(self, dat_file: str, log_file: str):
        self.dat_file = Path(dat_file)
        self.log_parser = LogParser(log_file)
        self.channel_map = self.log_parser.channel_map
        self.sample_size = len(self.channel_map)
        self.nb_samples = self.log_parser.metadata['nb_ech']

    def read_all_samples(self) -> Tuple[np.ndarray, Dict]:
        """Reads the entire file and reshapes based on channel map."""
        print(f"[DATParser] Nb_ech from log: {self.nb_samples}")
        print(f"[DATParser] Calculated sample_size (channels): {self.sample_size}")
        
        expected_size = self.nb_samples * self.sample_size
        print(f"[DATParser] Expected total int32 values: {expected_size}")
        
        # Read file
        with open(self.dat_file, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.int32)
            
        file_size = data.size
        print(f"[DATParser] Actual file int32 values: {file_size}")
        
        if file_size != expected_size:
             # Detailed debug of channel map
             mic_count = sum(1 for c in self.channel_map if c['type'] == 'mic')
             va_count = sum(1 for c in self.channel_map if c['type'] == 'va')
             ctr_count = sum(1 for c in self.channel_map if c['type'] == 'counter')
             print(f"[DATParser] Channel Breakdown: Mics={mic_count}, VAs={va_count}, Counters={ctr_count}, Total={len(self.channel_map)}")
             
             raise ValueError(f"File size mismatch: expected {expected_size} values "
                              f"({self.nb_samples} samples x {self.sample_size} channels), "
                              f"got {file_size}. Check log file definition.")
        
        # Reshape (Samples x Channels)
        reshaped_data = data.reshape((self.nb_samples, self.sample_size))
        return reshaped_data, self.log_parser.metadata

    def extract_microphone_signals(self) -> np.ndarray:
        """Extract only microphone columns."""
        data, _ = self.read_all_samples()
        mic_indices = [i for i, ch in enumerate(self.channel_map) if ch['type'] == 'mic']
        return data[:, mic_indices]

    def extract_analog_signals(self) -> np.ndarray:
        """Extract only analog columns."""
        data, _ = self.read_all_samples()
        va_indices = [i for i, ch in enumerate(self.channel_map) if ch['type'] == 'va']
        return data[:, va_indices]

    def extract_counters(self) -> np.ndarray:
        """Extract only counter columns."""
        data, _ = self.read_all_samples()
        ctr_indices = [i for i, ch in enumerate(self.channel_map) if ch['type'] == 'counter']
        return data[:, ctr_indices]

    def get_metadata(self) -> Dict:
        return self.log_parser.metadata

if __name__ == "__main__":
    # Test script
    import sys
    if len(sys.argv) > 1:
        fname = sys.argv[1]
        logname = fname.replace('.dat', '.log')
        try:
            parser = DATParser(fname, logname)
            print(f"Loaded {fname}")
            print(f"Channels: {parser.sample_size}")
            print(f"Samples: {parser.nb_samples}")
            
            mics = parser.extract_microphone_signals()
            print(f"Microphones shape: {mics.shape}")
            
            vas = parser.extract_analog_signals()
            print(f"Analog shape: {vas.shape}")

        except Exception as e:
            print(f"Error: {e}")
