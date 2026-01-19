import matplotlib.pyplot as plt
from matplotlib import ticker
import h5py as h5
import numpy as np
import scipy.signal as sig
import os
import glob
import warnings
from pathlib import Path
from DATParser import DATParser, LogParser
warnings.filterwarnings("ignore", category=RuntimeWarning) 
def on_press(event):
    print('you pressed', event.button, event.xdata, event.ydata)


def load_microphone_signals(file_path: str, temperature: float = 26) -> tuple:
    """
    Load microphone signals from HDF5 or DAT file.
    
    Args:
        file_path: Path to HDF5 or DAT file
        temperature: Temperature in Celsius (used for sound speed calculation)
    
    Returns:
        tuple: (mics_array, ref_signal, num_tixels, sound_speed)
    """
    file_path = str(file_path)
    
    # Determine sound speed based on temperature
    sound_speed = np.sqrt(1.4 * 287 * (temperature + 273))
    
    if file_path.lower().endswith('.h5') or file_path.lower().endswith('.hdf5'):
        # Load from HDF5
        data = h5.File(file_path, 'r')
        Secs = [int(i) for i in data['muh5'].keys()]
        NbSecs = len(data['muh5'].keys())
        
        Sig = np.concatenate([data['muh5'][str(i)]['sig'][:] for i in range(NbSecs)], axis=1)
        
        # Validate counter
        Cmptr = Sig[0, :]
        NbTixels = len(Cmptr)
        Chk = np.sum(np.diff(Cmptr))
        if Chk != NbTixels - 1:
            raise ValueError(f"Bad Counter in {file_path}")
        
        # Extract channels
        Mics = Sig[np.arange(1, 257), :]
        Mics = np.vstack((Mics, Sig[259, :]))
        Ref = -Sig[257, :]  # Volts
        
        data.close()
        
    elif file_path.lower().endswith('.dat'):
        # Load from DAT
        log_file = file_path.replace('.dat', '.log')
        if not os.path.exists(log_file):
            raise FileNotFoundError(f"Log file not found: {log_file}")
        
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
            print("Warning: No analog reference channel found, using first microphone")
        
        NbTixels = Mics.shape[0]
        
        # Update temperature from log metadata if available
        metadata = dat_parser.get_metadata()
        print(f"  DAT metadata: {metadata['nb_micros_actifs']} mics, {metadata['freq']} Hz")
        
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    return Mics, Ref, NbTixels, sound_speed



Fe = 50e3
dt = 1/Fe
Po = 20e-6

NbCmpt = 1
NbMems0 = 256
NbMics = 257
NbAnal = 1
SensMems = 3.54e-6#Pa/digital unit
SAna = 2**23/2.5 #sensibilite des voies analogiques

NbVoies = NbCmpt + NbAnal   + NbMems0
NumSnd = 257

PathCalib = '../DataCalib4'
Files = os.listdir(PathCalib)
NbFiles = len(Files)

Lmax = 3
lmax = 1.

plt.close('all')

# Process files
toa = []
for ff, File in enumerate(sorted(Files)):
    if File.startswith('.'):
        continue
    
    file_path = os.path.join(PathCalib, File)
    
    print(f"Processing {File}...")
    
    try:
        # Load signals (supports both HDF5 and DAT formats)
        Tc = 26  # Temperature in Celsius
        Mics, Ref, NbTixels, C = load_microphone_signals(file_path, Tc)
        
    except Exception as e:
        print(f"  Error: {e}")
        continue
    
    # Validate counter (for HDF5 format)
    if File.lower().endswith(('.h5', '.hdf5')):
        Cmptr = Ref  # Already validated in load_microphone_signals
    
    t = np.arange(NbTixels) / Fe
    tmax = Lmax / C
    tfmax = lmax / C
    ifmax = tfmax * Fe
    
    # TOA Extraction using GCC-PHAT
    NFFT = NbTixels
    df = Fe / NFFT
    T = np.zeros(NbMics)
    
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
    toa = np.append(toa, toas)
    fig = plt.figure(ff)
    plt.pcolormesh(range(NbMics), tt*C, GCCE.T, cmap='Greys')
    plt.plot(range(NbMics), np.array(toas)*C, '.r')
    plt.title(f'GCC-PHAT - {File}')
    plt.xlabel('Microphone')
    plt.ylabel('Distance (m)')
    plt.show()

# Reshape TOA matrix
toa = np.array(toa).reshape((-1, NbMics))

# Display results
fig = plt.figure(figsize=(12, 6))
plt.pcolormesh(toa)
plt.yticks(np.arange(0, NbMics, 8))
plt.colorbar(label='Time Delay (s)')
plt.title('TOA Matrix')
plt.xlabel('Microphone')
plt.ylabel('Acquisition #')
plt.grid()
plt.show()

# Save results
np.save('toaUp_violon.npy', toa)
print(f"\n✓ TOA extraction complete")
print(f"  Shape: {toa.shape}")
print(f"  Output: toaUp_violon.npy")