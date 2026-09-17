#%%
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.signal as sig
import GeoCalibUtils as gcu
from numpy.fft import rfft, irfft, fft, ifft

from tqdm import tqdm 
   
mpl.rcParams['lines.linewidth'] = 0.5

fin = "data/GeoCalib_00.dat"
NbMems = 480
Fe = 50000
NbVoies = NbMems+2
fout = fin[:-4]+'_'+'TDOA.npy'
    
#%%
# Channel order per tixel: counter, mems 0-255, analog (TTL), mems 256-479
NbMemsLow = 256
NbMemsHigh = NbMems - NbMemsLow  # 224
ChunkDuration = 0.1  # seconds
ChunkSamples = int(ChunkDuration * Fe)  # 5000 samples
AnalogCol = 1 + NbMemsLow

# Pass 1: memory-map the file and scan only the analog (TTL) channel to find
# rising-edge indices, without loading the mems channels into memory.
Mm = np.memmap(fin, dtype='int32', mode='r')
NbTixTotal = Mm.size // NbVoies
Mm = Mm[:NbTixTotal * NbVoies].reshape((NbTixTotal, NbVoies))
Analog = np.asarray(Mm[:, AnalogCol])

Thresh = (Analog.max().astype(np.float64) + Analog.min().astype(np.float64)) / 2
Above = Analog > Thresh
RisingEdges = np.where(Above[1:] & ~Above[:-1])[0] + 1

# Pass 2: load the mems channels only for each detected chunk, for the chunk duration
t = np.arange(ChunkSamples)/Fe
Spark = []
for ii in RisingEdges:
    if ii + ChunkSamples <= NbTixTotal:
        Chunk = np.asarray(Mm[ii:ii + ChunkSamples, :])  # Shape: (ChunkSamples, NbVoies)
        MemsLow = Chunk[:, 1:1 + NbMemsLow].T
        MemsHigh = Chunk[:, 2 + NbMemsLow:2 + NbMemsLow + NbMemsHigh].T
        Spark.append(np.vstack([MemsLow, MemsHigh]))  # Shape: (NbMems, ChunkSamples)
Spark = np.array(Spark)
NbSrcs = Spark.shape[0]
#_=plt.plot(t, Spark.reshape(-1,ChunkSamples).T)
#%%
from numpy.fft import rfftfreq, rfft, irfft
from concurrent.futures import ThreadPoolExecutor, as_completed

def local_maxima_mask(Wave):
    """Per-row boolean mask of local maxima in a circular (period nfftUp) signal."""
    Prev = np.roll(Wave, 1, axis=1)
    Next = np.roll(Wave, -1, axis=1)
    return (Wave > Prev) & (Wave > Next)


def theil_sen_trend(x, y):
    """
    Robust local trend through (x, y) via Theil-Sen (median of all pairwise
    slopes). Mems in a beam sit along a line, so their arrival times often
    ramp rather than plateau -- a flat median-of-neighbors mismodels that,
    and breaks down further once 2+ of the few points in a beam are bad.
    Theil-Sen tolerates close to half the points being outliers.
    """
    idx_i, idx_j = np.triu_indices(len(x), k=1)
    slopes = (y[idx_j] - y[idx_i]) / (x[idx_j] - x[idx_i])
    slope = np.median(slopes)
    intercept = np.median(y - slope * x)
    return intercept + slope * x


def block_smoothness_index(TauMat, BeamSize=8, eps=1e-12):
    """
    Mems are ordered in beams of BeamSize consecutive channels that sit only
    a few cm apart, so within any BeamSize x BeamSize block of Tau (one
    beam's rows against another beam's columns) the values should vary
    slowly. A rough block usually means one channel in that beam locked
    onto the wrong signal peak. Score in [0, 1], 1 when the second
    differences within blocks are small relative to Tau's overall scale.
    """
    N = TauMat.shape[0]
    NbBeamsSub = N // BeamSize
    if NbBeamsSub * BeamSize != N or BeamSize < 3:
        return 1.0
    T = TauMat.reshape(NbBeamsSub, BeamSize, NbBeamsSub, BeamSize)
    Roughness = np.linalg.norm(np.diff(T, n=2, axis=1)) + np.linalg.norm(np.diff(T, n=2, axis=3))
    Scale = np.linalg.norm(T) + eps
    return 1.0 / (1.0 + Roughness / Scale)


def tau_quality_index(TauMat, BeamSize=8, eps=1e-12):
    """
    Quality index in [0, 1] for a single-source TDOA matrix, based on how well
    it matches the theoretical structure Tau[m, n] = a_m - a_n expected for a
    single point source (a_m being the source-to-sensor m propagation time):
      - global symmetry: Tau must be skew-symmetric (Tau = -Tau.T)
      - global rank: a_m - a_n has rank <= 2, so Tau's skew-symmetric part
        should be a rank-2 matrix (2 dominant, equal singular values)
      - local smoothness: within any BeamSize x BeamSize block Tau should
        vary slowly (see block_smoothness_index)
    The three scores are combined multiplicatively so the index is 1 only
    when all properties hold.
    """
    Skew = (TauMat - TauMat.T) / 2
    Sym = (TauMat + TauMat.T) / 2
    skewEnergy = np.linalg.norm(Skew)
    symEnergy = np.linalg.norm(Sym)
    symmetryQ = skewEnergy / (skewEnergy + symEnergy + eps)

    s = np.linalg.svd(Skew, compute_uv=False)
    rankQ = (s[0] ** 2 + s[1] ** 2) / (np.sum(s ** 2) + eps)

    smoothnessQ = block_smoothness_index(TauMat, BeamSize, eps)

    return symmetryQ * rankQ * smoothnessQ

def compute_tau_source(SparkOne, t, NChunk=60, BeamSize=8, MinRelAmp=0.3, OutlierTol=5.0):
    """
    TDOA matrix (NbMems x NbMems) and quality index for a single source, from
    each channel's own impulse peak (IMP) rather than pairwise
    cross-correlation (GCC): each channel is upsampled once and its own
    argmax gives that channel's arrival time a_m, then Tau[m, n] = a_m - a_n.
    This is NbMems FFTs total instead of NbMems^2 cross-correlations.

    The upsampled ifft (NbMems x nfftUp, e.g. 480 x 50000 complex128 -- ~366
    MiB) is computed NChunk channels at a time instead of all NbMems at
    once: with many sources running concurrently in the thread pool, one
    full-width array per thread was enough to exhaust memory.

    Mems are ordered in beams of BeamSize consecutive channels only a few cm
    apart, so their arrival times should form a smooth local trend (often a
    ramp, since a beam is a straight run of mems and a source isn't
    generally equidistant to all of them) -- fitted per beam with Theil-Sen,
    robust to several simultaneously-bad channels in the same beam. When a
    channel's argmax pick disagrees with that trend by more than OutlierTol
    times the beam's residual spread, earlier (still significant) local
    maxima in that channel's own upsampled signal are tried instead
    (recomputed on demand, so the full NbMems x nfftUp array is never held
    at once), keeping whichever candidate best agrees with the trend.
    OutlierTol=5 (raw MAD multiples here) matches the standard ~3.5
    modified-z-score robust outlier convention.
    """
    nfft = SparkOne.shape[1]
    Up = 10
    nfftUp = int(Up * nfft)
    dtUp = t[1] / Up
    tUp = np.arange(nfftUp) * dtUp
    half = nfftUp / 2

    def idx_to_time(idx):
        return np.where(idx < half, tUp[idx], tUp[idx] - tUp[-1])

    SpAll = fft(SparkOne, axis=1)   # (NbMems, nfft), one FFT per channel

    a = np.zeros(NbMems)
    for n0 in range(0, NbMems, NChunk):
        n1 = min(n0 + NChunk, NbMems)
        Wave = np.real(ifft(SpAll[n0:n1], axis=1, n=nfftUp))
        imax = np.argmax(Wave, axis=1)
        a[n0:n1] = idx_to_time(imax)

    NbBeams = NbMems // BeamSize
    if NbBeams * BeamSize == NbMems and BeamSize >= 3:
        xBeam = np.arange(BeamSize, dtype=float)
        for b in range(NbBeams):
            idxBeam = np.arange(b * BeamSize, (b + 1) * BeamSize)
            trend = theil_sen_trend(xBeam, a[idxBeam])
            resid = a[idxBeam] - trend
            spread = np.median(np.abs(resid)) + 1e-9
            for k, m in enumerate(idxBeam):
                if abs(resid[k]) <= OutlierTol * spread:
                    continue  # not an outlier relative to the beam's local trend

                WaveM = np.real(ifft(SpAll[m], n=nfftUp))
                imaxM = np.argmax(WaveM)
                PeakM = WaveM[imaxM]
                MaskM = (WaveM > np.roll(WaveM, 1)) & (WaveM > np.roll(WaveM, -1))
                cand = np.where(MaskM)[0]
                cand = cand[cand < imaxM]  # earlier peaks only
                cand = cand[WaveM[cand] >= MinRelAmp * PeakM]  # still significant
                if cand.size == 0:
                    continue

                cand_a = idx_to_time(cand)
                best = np.argmin(np.abs(cand_a - trend[k]))
                if np.abs(cand_a[best] - trend[k]) < abs(resid[k]):
                    a[m] = cand_a[best]

    TauMat = a[:, None] - a[None, :]
    Quality = tau_quality_index(TauMat, BeamSize)
    return TauMat, Quality, a

nfft = Spark.shape[2]
df = Fe / nfft

# Process sources in random order. Each worker computes both the TDOA matrix
# AND its quality index (the selection process), so the main thread only has
# to look at the returned score. As soon as enough sources (NbGoodTarget)
# reach the best quality index found so far, the remaining, not-yet-started
# tasks are cancelled and the pool is torn down without waiting for them.
NbGoodTarget = 20
QualTol = 0.10  # a source is "good" once its index is within 10% of the running best

Order = np.random.permutation(NbSrcs)
OrderPos = {s: i for i, s in enumerate(Order)}  # source index -> position in the random draw order
TauAll = {}
ToaAll = {}
QualAll = {}
QMax = 0.0

MaxWorkers = min(8, os.cpu_count() or 4)  # bounds peak memory: several 480x50000 ifft chunks can be live at once
with ThreadPoolExecutor(max_workers=MaxWorkers) as executor:
    futures = {executor.submit(compute_tau_source, Spark[s], t): s for s in Order}
    pbar = tqdm(total=len(futures), desc="Sources processed")
    try:
        for future in as_completed(futures):
            s = futures[future]
            TauMat, Q, ToaVec = future.result()
            TauAll[s] = TauMat
            ToaAll[s] = ToaVec
            QualAll[s] = Q
            QMax = max(QMax, Q)
            GoodCount = sum(1 for q in QualAll.values() if q >= QMax * (1 - QualTol))

            pbar.update(1)
            pbar.set_postfix(src=s, order=OrderPos[s], best=f"{QMax:.4f}", good=GoodCount)
            if GoodCount >= NbGoodTarget:
                break
    finally:
        for f in futures:
            f.cancel()  # no-op for futures already running/done
        pbar.close()

SrcAll = np.array(list(TauAll.keys()))
QualAll = np.array([QualAll[s] for s in SrcAll])
TauAll = np.array([TauAll[s] for s in SrcAll])
ToaAll = np.array([ToaAll[s] for s in SrcAll])
BestMask = QualAll >= QMax * (1 - QualTol)
print(f"Processed {len(SrcAll)}/{NbSrcs} sources, "
      f"{BestMask.sum()} reached best quality index {QMax:.4f}")

# Keep only the sources whose TDOA matrix reached the best quality index
Tau = TauAll[BestMask]
Toa = ToaAll[BestMask]  # (NbGoodSrcs, NbMems), each row the per-channel arrival times behind that source's Tau
SrcBest = SrcAll[BestMask]
QualBest = QualAll[BestMask]
#%%
NbSrcs = Tau.shape[0]
fig = gcu.PlotTau(Tau, NbSrcs)
plt.show(block = True)
np.save(fout, Tau)
np.save(fin[:-4] + '_TOA.npy', Toa)