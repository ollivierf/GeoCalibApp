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

def tau_quality_index(TauMat, eps=1e-12):
    """
    Quality index in [0, 1] for a single-source TDOA matrix, based on how well
    it matches the theoretical structure Tau[m, n] = a_m - a_n expected for a
    single point source (a_m being the source-to-sensor m propagation time):
      - global symmetry: Tau must be skew-symmetric (Tau = -Tau.T)
      - global rank: a_m - a_n has rank <= 2, so Tau's skew-symmetric part
        should be a rank-2 matrix (2 dominant, equal singular values)
    The two scores are combined multiplicatively so the index is 1 only when
    both properties hold.
    """
    Skew = (TauMat - TauMat.T) / 2
    Sym = (TauMat + TauMat.T) / 2
    skewEnergy = np.linalg.norm(Skew)
    symEnergy = np.linalg.norm(Sym)
    symmetryQ = skewEnergy / (skewEnergy + symEnergy + eps)

    s = np.linalg.svd(Skew, compute_uv=False)
    rankQ = (s[0] ** 2 + s[1] ** 2) / (np.sum(s ** 2) + eps)

    return symmetryQ * rankQ

def compute_tau_source(SparkOne, t, NChunk=60):
    """
    TDOA matrix (NbMems x NbMems) and quality index for a single source.
    The channel FFT is computed once per channel (not once per pair), and
    each row m is obtained by vectorized cross-correlation against every
    other channel n -- the selection (quality) itself is evaluated here,
    inside the worker, so the caller only needs the outcome.

    The upsampled ifft (NbMems x nfftUp, e.g. 480 x 50000 complex128 -- ~366
    MiB) is computed NChunk channels at a time instead of all NbMems at
    once: with many sources running concurrently in the thread pool, one
    full-width array per thread was enough to exhaust memory.
    """
    nfft = SparkOne.shape[1]
    Up = 10
    nfftUp = int(Up * nfft)
    dtUp = t[1] / Up
    tUp = np.arange(nfftUp) * dtUp
    half = nfftUp / 2

    SpAll = fft(SparkOne, axis=1)   # (NbMems, nfft), one FFT per channel
    AbsAll = np.abs(SpAll)

    TauMat = np.zeros((NbMems, NbMems))
    for m in range(NbMems):
        for n0 in range(0, NbMems, NChunk):
            n1 = min(n0 + NChunk, NbMems)
            SX = np.conj(SpAll[m]) * SpAll[n0:n1] / (AbsAll[m] * AbsAll[n0:n1] + 1e-12)
            G = np.real(ifft(SX, axis=1, n=nfftUp))
            imax = np.argmax(G, axis=1)
            TauMat[n0:n1, m] = np.where(imax < half, tUp[imax], tUp[imax] - tUp[-1])

    Quality = tau_quality_index(TauMat)
    return TauMat, Quality

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
QualAll = {}
QMax = 0.0

MaxWorkers = min(8, os.cpu_count() or 4)  # bounds peak memory: several 480x50000 ifft chunks can be live at once
with ThreadPoolExecutor(max_workers=MaxWorkers) as executor:
    futures = {executor.submit(compute_tau_source, Spark[s], t): s for s in Order}
    pbar = tqdm(total=len(futures), desc="Sources processed")
    try:
        for future in as_completed(futures):
            s = futures[future]
            TauMat, Q = future.result()
            TauAll[s] = TauMat
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
BestMask = QualAll >= QMax * (1 - QualTol)
print(f"Processed {len(SrcAll)}/{NbSrcs} sources, "
      f"{BestMask.sum()} reached best quality index {QMax:.4f}")

# Keep only the sources whose TDOA matrix reached the best quality index
Tau = TauAll[BestMask]
SrcBest = SrcAll[BestMask]
QualBest = QualAll[BestMask]
#%%
NbSrcs = Tau.shape[0]
fig = gcu.PlotTau(Tau, NbSrcs)
plt.show(block = True)
np.save(fout, Tau)
