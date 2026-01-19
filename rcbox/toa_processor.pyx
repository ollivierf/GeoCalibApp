# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
import scipy.signal as sig

def compute_toa_cython(np.ndarray[np.float64_t, ndim=2] Mics, 
                       np.ndarray[np.float64_t, ndim=1] Ref, 
                       double Fe,
                       double Tc,
                       double C,
                       bint return_gcc_data=False):
    """
    Compute TOA using GCC-PHAT algorithm with Cython optimization.
    
    Parameters:
        Mics (np.ndarray): Microphone signals (NbMics, NbSamples)
        Ref (np.ndarray): Reference signal (NbSamples,)
        Fe (float): Sampling frequency
        C (float): Speed of sound
        return_gcc_data (bool): Whether to return GCC validation data
        
    Returns:
        toas (np.ndarray): Computed TOA values
        (Optional) gcc_data (tuple): (GCCE, tt_limited, toas, imax, Tc, C)
    """
    cdef int NbTixels = Mics.shape[1]
    cdef int NFFT = NbTixels
    cdef double df = Fe / NFFT
    cdef double tmax = 10.0 / C
    cdef int NUp = 10 * NFFT
    cdef double dtUp = 1.0 / (NUp * df)
    
    # 1. FFTs (Numpy is already fast here, backed by C libraries)
    # Using np.fft.rfft for real input
    cdef np.ndarray[np.complex128_t, ndim=1] SRef = np.fft.rfft(Ref, NFFT)
    cdef np.ndarray[np.complex128_t, ndim=2] SMics = np.fft.rfft(Mics, NFFT)
    
    # 2. Generalized Cross Correlation with PHAT weighting
    # GCS = conj(SRef) * SMics / (|SMics| * |SRef|)
    # Note: Adding small epsilon to denominator to avoid division by zero might be good practice,
    # but strictly following original code.
    
    # Broadcasting SRef to match SMics shape
    cdef np.ndarray[np.complex128_t, ndim=2] SRef_conj = np.conj(SRef)[None, :]
    cdef np.ndarray[np.float64_t, ndim=2] AbsSMics = np.abs(SMics)
    cdef np.ndarray[np.float64_t, ndim=2] AbsSRef = np.abs(SRef)[None, :]
    
    # GCC-PHAT in Frequency Domain
    cdef np.ndarray[np.complex128_t, ndim=2] GCS = SRef_conj * SMics / (AbsSMics * AbsSRef + 1e-15)
    
    # 3. Inverse FFT to get Time Domain GCC
    cdef np.ndarray[np.float64_t, ndim=2] GCC = np.fft.irfft(GCS, NUp)
    cdef np.ndarray[np.float64_t, ndim=1] tt = np.arange(NUp) * dtUp
    
    # 4. Limit to 10 meters and compute envelope
    cdef np.ndarray[np.npy_bool, ndim=1] valid_idx_mask = tt < tmax
    
    # Get the valid slice of GCC
    # We need to compute indices for slicing
    cdef int valid_len = np.sum(valid_idx_mask)
    cdef np.ndarray[np.float64_t, ndim=2] GCC_valid = GCC[:, :valid_len]
    
    # Hilbert Transform for Envelope (scipy.signal.hilbert is efficient)
    cdef np.ndarray[np.float64_t, ndim=2] GCCE = np.abs(sig.hilbert(GCC_valid, axis=1))
    cdef np.ndarray[np.float64_t, ndim=1] tt_limited = tt[:valid_len]
    
    # 5. Peak Finding
    cdef int NbMics = GCCE.shape[0]
    cdef np.ndarray[np.int64_t, ndim=1] imax = np.zeros(NbMics, dtype=np.int64)
    cdef int i
    cdef np.ndarray[np.float64_t, ndim=1] signal_slice
    
    # Optimization: Loop over mics
    for i in range(NbMics):
        signal_slice = GCCE[i, :]
        
        # Using scipy.signal.find_peaks
        peaks, properties = sig.find_peaks(signal_slice, height=0)
        
        if len(peaks) > 0:
            peak_heights = properties['peak_heights']
            
            # Sort by height (descending)
            if len(peaks) > 1:
                # argsort returns indices that would sort the array
                # [::-1] reverses it for descending order
                sorted_indices = np.argsort(peak_heights)[::-1]
                
                # Take top 10
                n_top = min(10, len(peaks))
                top_indices = sorted_indices[:n_top]
                top_peaks = peaks[top_indices]
                
                # Take earliest (min index) of top peaks
                imax[i] = np.min(top_peaks)
            else:
                imax[i] = peaks[0]
        else:
            # Fallback
            imax[i] = np.argmax(signal_slice)
            
    # 6. Compute TOAs
    # Vectorized indexing
    # We need to clamp indices to be safe, though logic above should produce valid indices within 0..valid_len-1
    cdef np.ndarray[np.int64_t, ndim=1] safe_imax = np.minimum(imax, valid_len - 1)
    cdef np.ndarray[np.float64_t, ndim=1] toas = tt_limited[safe_imax]
    
    if return_gcc_data:
        # Calculate Tc from C roughly if needed, but we passed C. 
        # The caller passed Tc, so we can return it.
        # But wait, the function signature has C, not Tc.
        # Let's assume Tc was used to calc C.
        return toas, (GCCE, tt_limited, toas, imax, Tc, C)
    
    return toas
