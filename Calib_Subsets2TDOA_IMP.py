#%%
# Splits the 480 mems into overlapping subsets of "beams" (groups of 8
# successive mems) so the DTOA process can run on manageable chunks instead
# of all 480 channels at once. 60 beams total; 4 are drawn once and shared by
# every subset, the other 56 are partitioned (no overlap) into NbSubsets
# groups as evenly as possible. Each subset then gets its own Tau matrix,
# computed and saved independently.
import os
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import GeoCalibUtils as gcu
from numpy.fft import fft, ifft
from scipy.signal import hilbert
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

mpl.rcParams['lines.linewidth'] = 0.5

fin = "data/GeoCalib_00.dat"
OutDir = "results/subsets"
NbMems = 480
BeamSize = 8
NbBeams = NbMems // BeamSize      # 60
# Bigger, more overlapping subsets: the geometry step reconstructs each
# subset independently before merging, and a small subset (~10 beams) gives
# the RMDU solver too few redundant distance constraints to pin down an
# accurate rigid shape -- and too few common beams for the first-processed
# subset to reliably tell which of its beams sit on the cube's top face vs
# its side faces (expected ~20% top; a 10-beam sample can easily miss that).
# More common beams (shared by every subset) and fewer, larger subsets both
# push in the same direction: better-conditioned per-subset reconstructions.
NbCommonBeams = 8
NbSubsets = 5
Fe = 50000
NbVoies = NbMems + 2
Seed = None                       # set an int for a reproducible split

# Channel order per tixel: counter, mems 0-255, analog (TTL), mems 256-479
NbMemsLow = 256
ChunkDuration = 0.1  # seconds
ChunkSamples = int(ChunkDuration * Fe)  # 5000 samples
AnalogCol = 1 + NbMemsLow

NbGoodTarget = 12
QualTol = 0.05  # a source is "good" once its index is within 5% of the running best

os.makedirs(OutDir, exist_ok=True)


def mem_to_col(mem_idx):
    """Global mem index (0..NbMems-1) -> its column in the raw tixel file."""
    mem_idx = np.asarray(mem_idx)
    return np.where(mem_idx < NbMemsLow, mem_idx + 1, mem_idx + 2)


#%%
# ---- Build the beam -> mem table and draw the subsets ----
rng = np.random.default_rng(Seed)
Beams = [list(range(b * BeamSize, b * BeamSize + BeamSize)) for b in range(NbBeams)]

BeamOrder = rng.permutation(NbBeams)
CommonBeams = sorted(BeamOrder[:NbCommonBeams].tolist())
RemainingBeams = BeamOrder[NbCommonBeams:]

# Split the remaining beams into NbSubsets non-overlapping groups, as evenly
# as possible (e.g. 56 beams / 9 subsets -> two subsets get 7, seven get 6).
NbRemaining = len(RemainingBeams)
BaseSize, Extra = divmod(NbRemaining, NbSubsets)
SubsetSizes = [BaseSize + 1] * Extra + [BaseSize] * (NbSubsets - Extra)
rng.shuffle(SubsetSizes)

Subsets = []
cursor = 0
for size in SubsetSizes:
    OwnBeams = sorted(RemainingBeams[cursor:cursor + size].tolist())
    cursor += size
    BeamsHere = sorted(CommonBeams + OwnBeams)
    MemsHere = sorted(m for b in BeamsHere for m in Beams[b])
    Subsets.append({"beams": BeamsHere, "own_beams": OwnBeams, "mems": MemsHere})

with open(os.path.join(OutDir, "subset_manifest.json"), "w") as f:
    json.dump({"common_beams": CommonBeams, "subsets": Subsets}, f, indent=2)

print(f"{NbBeams} beams total, {len(CommonBeams)} common to all subsets, "
      f"{NbSubsets} subsets covering the remaining {NbRemaining} beams "
      f"(sizes {sorted(SubsetSizes)}).")

#%%
# ---- Pass 1: detect sparks once from the analog (TTL) channel, shared by
# every subset (rising edges don't depend on which mems are kept). ----
Mm = np.memmap(fin, dtype='int32', mode='r')
NbTixTotal = Mm.size // NbVoies
Mm = Mm[:NbTixTotal * NbVoies].reshape((NbTixTotal, NbVoies))
Analog = np.asarray(Mm[:, AnalogCol])

Thresh = (Analog.max().astype(np.float64) + Analog.min().astype(np.float64)) / 2
Above = Analog > Thresh
RisingEdges = np.where(Above[1:] & ~Above[:-1])[0] + 1
RisingEdges = RisingEdges[RisingEdges + ChunkSamples <= NbTixTotal]
del Analog, Above

t = np.arange(ChunkSamples) / Fe


#%%
def rising_edges_mask(mask):
    """Per-row boolean mask of samples where a boolean row goes False->True."""
    Prev = np.pad(mask[:, :-1], ((0, 0), (1, 0)), constant_values=False)
    return mask & ~Prev


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


def compute_tau_source(SparkOne, t, BeamSize=8, OutlierTol=5.0,
                        CrossBeamOutlierTol=5.0, MaxBadBeamFrac=0.3,
                        SearchWindowSec=0.03, NoiseWindowSec=0.003,
                        RelOnsetFrac=0.15, NoiseFloorMult=6.0):
    """
    TDOA matrix (NbMemsSub x NbMemsSub) and quality index for a single
    source, restricted to the mems in this subset. Uses each channel's own
    impulse arrival (IMP) rather than pairwise cross-correlation (GCC).

    The recorded signal is not one clean broadband pulse: it's a sequence of
    several comparable-amplitude echo/reverberation bursts spread across the
    chunk (structure ringing), so a plain argmax of the raw waveform picks
    whichever burst happens to contain the single tallest sample -- which
    can differ between channels of the very same beam for the same physical
    arrival, and even between cycles of one burst (its envelope has several
    similarly-tall oscillations near the top). Each channel's arrival time
    a_m is instead the *onset* of its analytic-signal envelope: the first
    point, within SearchWindowSec of the chunk start (comfortably covering
    every direct-path arrival on this rig, and well short of the next, much
    weaker reverberation burst around 35-40ms), where the envelope crosses
    max(RelOnsetFrac fraction of that channel's own peak in the window,
    NoiseFloorMult times the noise floor measured from the known-quiet
    first NoiseWindowSec). Then Tau[m, n] = a_m - a_n.

    Mems are ordered in beams of BeamSize consecutive channels only a few cm
    apart, so their arrival times should form a smooth local trend (often a
    ramp, since a beam is a straight run of mems and a source isn't
    generally equidistant to all of them) -- fitted per beam with Theil-Sen,
    robust to several simultaneously-bad channels in the same beam. When a
    channel's onset pick disagrees with that trend by more than OutlierTol
    times the beam's residual spread, its other envelope threshold
    crossings in the search window (e.g. a later, stronger burst if the
    first one was too weak on this channel) are tried instead, keeping
    whichever candidate best agrees with the trend. OutlierTol=5 (raw MAD
    multiples here) matches the standard ~3.5 modified-z-score robust
    outlier convention.

    That per-beam correction only compares a beam's channels against each
    other, so it can't catch every channel in a beam locking onto the same
    wrong peak together (e.g. a reflection mistaken for the direct path) --
    the beam still looks internally smooth, just offset from the rest of
    the array. A second, cross-beam check catches that: each beam's median
    arrival time is compared to the array-wide median of all beams' medians
    (for this source), and a beam more than CrossBeamOutlierTol times the
    robust spread away is masked out (NaN) rather than left to silently
    corrupt the reconstructed geometry. If more than MaxBadBeamFrac of the
    source's beams are implicated, the whole source is untrustworthy and
    its quality is zeroed instead (excluding it from selection).
    """
    NbMemsSub = SparkOne.shape[0]
    nfft = SparkOne.shape[1]
    Up = 10
    nfftUp = int(Up * nfft)
    dtUp = t[1] / Up
    tUp = np.arange(nfftUp) * dtUp
    half = nfftUp / 2

    def idx_to_time(idx):
        return np.where(idx < half, tUp[idx], tUp[idx] - tUp[-1])

    SpAll = fft(SparkOne, axis=1)
    Wave = np.real(ifft(SpAll, axis=1, n=nfftUp))  # (NbMemsSub, nfftUp)
    Env = np.abs(hilbert(Wave, axis=1))

    win = max(int(round(SearchWindowSec / dtUp)), 1)
    noise_n = max(int(round(NoiseWindowSec / dtUp)), 1)
    EnvWin = Env[:, :win]
    NoiseFloor = np.median(EnvWin[:, :noise_n], axis=1)
    Thresh = np.maximum(RelOnsetFrac * EnvWin.max(axis=1), NoiseFloorMult * NoiseFloor) + 1e-12

    Rising = rising_edges_mask(EnvWin >= Thresh[:, None])
    imax = np.array([
        np.where(Rising[m])[0][0] if Rising[m].any() else int(np.argmax(EnvWin[m]))
        for m in range(NbMemsSub)
    ])
    a = idx_to_time(imax)

    NbBeamsSub = NbMemsSub // BeamSize
    if NbBeamsSub * BeamSize == NbMemsSub and BeamSize >= 3:
        xBeam = np.arange(BeamSize, dtype=float)
        for b in range(NbBeamsSub):
            idxBeam = np.arange(b * BeamSize, (b + 1) * BeamSize)
            trend = theil_sen_trend(xBeam, a[idxBeam])
            resid = a[idxBeam] - trend
            spread = np.median(np.abs(resid)) + 1e-9
            for k, m in enumerate(idxBeam):
                if abs(resid[k]) <= OutlierTol * spread:
                    continue  # not an outlier relative to the beam's local trend

                cand = np.where(Rising[m])[0]  # this channel's other threshold crossings
                if cand.size == 0:
                    continue

                cand_a = idx_to_time(cand)
                best = np.argmin(np.abs(cand_a - trend[k]))
                if np.abs(cand_a[best] - trend[k]) < abs(resid[k]):
                    a[m] = cand_a[best]

    TauMat = a[:, None] - a[None, :]
    Quality = tau_quality_index(TauMat, BeamSize)

    if NbBeamsSub * BeamSize == NbMemsSub and NbBeamsSub >= 3:
        beam_medians = np.array([np.median(a[b * BeamSize:(b + 1) * BeamSize]) for b in range(NbBeamsSub)])
        array_med = np.median(beam_medians)
        beam_spread = np.median(np.abs(beam_medians - array_med)) + 1e-9
        bad_beams = np.abs(beam_medians - array_med) > CrossBeamOutlierTol * beam_spread
        if bad_beams.any():
            if bad_beams.mean() > MaxBadBeamFrac:
                Quality = 0.0  # too much of this source is corrupted to trust any of it
            else:
                for b in np.where(bad_beams)[0]:
                    idx = slice(b * BeamSize, (b + 1) * BeamSize)
                    a[idx] = np.nan
                    TauMat[idx, :] = np.nan
                    TauMat[:, idx] = np.nan

    return TauMat, Quality, a


def extract_spark_chunks(cols):
    """
    Reads each detected spark chunk (full channel width, transiently) and
    keeps only the given columns, so the accumulated Spark array is sized for
    this subset's mems only, not all 480.
    """
    Spark = []
    for ii in RisingEdges:
        Chunk = np.asarray(Mm[ii:ii + ChunkSamples, :])  # (ChunkSamples, NbVoies)
        Spark.append(Chunk[:, cols].T)  # (NbMemsSub, ChunkSamples)
    return np.array(Spark)


def process_subset(idx, subset):
    mems = subset["mems"]
    cols = mem_to_col(mems)
    Spark = extract_spark_chunks(cols)
    NbSrcs = Spark.shape[0]

    Order = np.random.permutation(NbSrcs)
    OrderPos = {s: i for i, s in enumerate(Order)}
    TauAll = {}
    ToaAll = {}
    QualAll = {}
    QMax = 0.0

    MaxWorkers = min(8, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=MaxWorkers) as executor:
        futures = {executor.submit(compute_tau_source, Spark[s], t): s for s in Order}
        pbar = tqdm(total=len(futures), desc=f"Subset {idx:02d} sources")
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
                f.cancel()
            pbar.close()

    SrcAll = np.array(list(TauAll.keys()))
    QualAll = np.array([QualAll[s] for s in SrcAll])
    TauAll = np.array([TauAll[s] for s in SrcAll])
    ToaAll = np.array([ToaAll[s] for s in SrcAll])
    BestMask = QualAll >= QMax * (1 - QualTol)
    print(f"Subset {idx:02d}: processed {len(SrcAll)}/{NbSrcs} sources, "
          f"{BestMask.sum()} reached best quality index {QMax:.4f} "
          f"({len(mems)} mems)")

    Tau = TauAll[BestMask]
    Toa = ToaAll[BestMask]  # (NbGoodSrcs, NbMemsSub), each row the per-channel arrival times behind that source's Tau
    fout = os.path.join(OutDir, f"subset_{idx:02d}_TDOA.npy")
    np.save(fout, Tau)
    np.save(os.path.join(OutDir, f"subset_{idx:02d}_TOA.npy"), Toa)

    fig, _, _ = gcu.PlotTau(Tau, Tau.shape[0])
    fig.savefig(os.path.join(OutDir, f"subset_{idx:02d}_TDOA.png"))
    plt.close(fig)

    return fout


#%%
# ---- Run the DTOA process independently for each subset ----
OutFiles = []
for idx, subset in enumerate(Subsets):
    OutFiles.append(process_subset(idx, subset))

print("Done:")
for f in OutFiles:
    print(" ", f)
