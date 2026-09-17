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

NbGoodTarget = 6
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
    TDOA matrix (NbMemsSub x NbMemsSub) and quality index for a single source,
    restricted to the mems in this subset. Same approach as the full-array
    version: one FFT per channel, vectorized cross-correlation row by row.
    """
    NbMemsSub = SparkOne.shape[0]
    nfft = SparkOne.shape[1]
    Up = 10
    nfftUp = int(Up * nfft)
    dtUp = t[1] / Up
    tUp = np.arange(nfftUp) * dtUp
    half = nfftUp / 2

    SpAll = fft(SparkOne, axis=1)
    AbsAll = np.abs(SpAll)

    TauMat = np.zeros((NbMemsSub, NbMemsSub))
    for m in range(NbMemsSub):
        for n0 in range(0, NbMemsSub, NChunk):
            n1 = min(n0 + NChunk, NbMemsSub)
            SX = np.conj(SpAll[m]) * SpAll[n0:n1] / (AbsAll[m] * AbsAll[n0:n1] + 1e-12)
            G = np.real(ifft(SX, axis=1, n=nfftUp))
            imax = np.argmax(G, axis=1)
            TauMat[n0:n1, m] = np.where(imax < half, tUp[imax], tUp[imax] - tUp[-1])

    Quality = tau_quality_index(TauMat)
    return TauMat, Quality


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
    QualAll = {}
    QMax = 0.0

    MaxWorkers = min(8, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=MaxWorkers) as executor:
        futures = {executor.submit(compute_tau_source, Spark[s], t): s for s in Order}
        pbar = tqdm(total=len(futures), desc=f"Subset {idx:02d} sources")
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
                f.cancel()
            pbar.close()

    SrcAll = np.array(list(TauAll.keys()))
    QualAll = np.array([QualAll[s] for s in SrcAll])
    TauAll = np.array([TauAll[s] for s in SrcAll])
    BestMask = QualAll >= QMax * (1 - QualTol)
    print(f"Subset {idx:02d}: processed {len(SrcAll)}/{NbSrcs} sources, "
          f"{BestMask.sum()} reached best quality index {QMax:.4f} "
          f"({len(mems)} mems)")

    Tau = TauAll[BestMask]
    fout = os.path.join(OutDir, f"subset_{idx:02d}_TDOA.npy")
    np.save(fout, Tau)

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
