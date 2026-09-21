#%%
# End-to-end pipeline, raw tixel spark data -> complete measured array geometry:
#   Stage 1 (per subset, in order): extract each subset's TDOA/TOA matrices
#     from the raw spark chunks (impulse-onset method, see compute_tau_source),
#     searching sources until NbGoodTarget reach the best quality index found
#     so far. Runs for every subset before Stage 2 starts, since the
#     trigger-to-emission timing calibration below needs every subset's TOA
#     data at once.
#   Stage 2 (per subset, in order): two-step RMDU geometry reconstruction
#     (see run_rmdu) on that TOA data, merged one subset at a time onto a
#     single shared cube template (see merge_subset_into_cube).
# Both stages plot live: Stage 1 redraws the current best-quality source's
# Tau/Toa whenever a new best is found; Stage 2 redraws the subset's RMDU
# solve as it converges and the merged (measured) array as it's built up.
# Replaces the old two-script flow (Calib_Subsets2TDOA_IMP.py piping its
# results/subsets/*.npy to Calib_Subsets2Geometry.py via disk) with one
# process that keeps everything in memory between the two stages -- the
# per-subset .npy/.png artifacts are still written for inspection, but
# nothing is re-read back off disk to drive the geometry stage.
import os
import json
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import GeoCalibUtils as gcu
from numpy.fft import fft, ifft
from scipy.signal import hilbert
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from scipy.spatial.distance import squareform, pdist
from scipy.spatial import ConvexHull
from scipy.optimize import linear_sum_assignment
try:
    from scipy.spatial import QhullError
except ImportError:
    from scipy.spatial.qhull import QhullError

from rcbox.rmds import compute_Lpinv
from Calib_Geometry2Browser import build_comparison_html

# ---- Raw data / subset split / TDOA extraction ----
fin = "data/GeoCalib_00.dat"
SubsetsDir = "results/subsets"
GeometryDir = "results/geometry"
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
Seed = 0                           # fixes the subset split AND the per-subset source search
# order (see process_subset's rng) so runs are reproducible -- with Seed=None every run drew
# a different random beam-to-subset partition and a different set of 12 "good" sources, which
# made it impossible to tell whether a change to the solver helped or just got an easier draw
# (confirmed a problem this session: deviation stats swung between runs by more than some of
# the fixes being tested for). Set back to None for production runs once the solver itself is
# no longer under active investigation.

# Channel order per tixel: counter, mems 0-255, analog (TTL), mems 256-479
NbMemsLow = 256
ChunkDuration = 0.1  # seconds
ChunkSamples = int(ChunkDuration * Fe)  # 5000 samples
AnalogCol = 1 + NbMemsLow

NbGoodTarget = 12
QualTol = 0.05  # a source is "good" once its index is within 5% of the running best

# ---- Geometry reconstruction ----
BeamSpacing = 0.23   # m, mandatory pitch between consecutive mics of a beam (beam length = (BeamSize-1)*BeamSpacing = 1.61 m)
BeamAlignTol = 0.005  # m, mandatory but allows this much slack off the ideal collinear/regularly-spaced beam shape

# Known physical facts about the rig, checked against every beam right after it's
# merged (see check_beam_geometry): no mic's raw RMDU solve should ever sit farther
# than DeviationTolM from its template-snapped position, and every beam's own fitted
# direction should agree with its assigned slot's canonical direction (the template
# only has 2: horizontal for the top face, vertical for the side faces) to within
# BeamDirTolDeg -- vertical beams are grossly parallel to every other vertical beam
# even across different (perpendicular) side faces, since "vertical" is a single
# shared direction regardless of which face it's on; only top-vs-side beams are
# perpendicular. A violation flags a suspect beam (bad slot assignment or bad TOA
# data for that beam) rather than a real physical deviation, since neither should
# happen on the real rig.
DeviationTolM = 0.20   # m
BeamDirTolDeg = 15.0   # deg

Lambda1 = 0.05      # step 1: rough geometry
Lambda2 = 2.0        # step 2: refined geometry
MaxIter = 10000
EpsLimit = 1e-10
TempCelsius = 26
Ndim = 3

# m, sources capped within this distance from the mic centroid -- kept only as
# project_sources_outside's own default; every call site in run_rmdu now passes
# max_dist=0 to disable it. Measured directly from the real rig (well-converged
# unconstrained RMDU fit, subset 04): every one of the 12 real sources sits
# 3.4-4.7m from the mic centroid, so a 3.0m cap forced project_sources_outside to
# clip every single source inward every iteration -- that chronic distortion of
# the source positions was propagating through the Guttman-transform data-fit
# step into a systematic flattening of the recovered mic array (PCA
# singular-value ratio [1, 0.70, 0.39] instead of the data's true [1, 0.83,
# 0.69]). The same failure mode, generalized (stacking any of these hard
# constraints fights the data and each other), is why every constraint except
# the mic-hull push in project_sources_outside was removed from run_rmdu this
# session -- see its docstring.
MaxSourceDist = 5.0

# Runtime, not physical: a run counts as stalled (and stops early rather than
# burning the rest of max_iter) once eps stops improving meaningfully -- with
# the hard constraints above, eps typically settles into a stable plateau well
# above eps_limit rather than ever reaching it (observed: identical eps for
# 8000+ iterations straight before this was added).
StallWindow = 200
StallTol = 1e-3

# The physical mount: a CubeSide-m cube with no bottom face. The top face is
# horizontal, carrying BeamsPerFace beams parallel to one horizontal axis;
# each of the 4 vertical side faces carries BeamsPerFace vertical beams. On
# every face the beams are a BeamPitch-spaced, centered row (the mics along
# each beam already give the other grid pitch, BeamSpacing -- together they
# draw each face's mandatory BeamPitch x (BeamSize-1)*BeamSpacing grid, e.g.
# 0.145 x 1.61 m, up to the installation's own planar distortion). This
# gives exactly two possible beam directions in the whole array -- vertical
# (side faces) and horizontal (top face) -- which is what makes the per-beam
# direction alone enough to tell top-face from side-face beams (see
# discover_cube_frame).
CubeSide = 2.0
BeamsPerFace = 12
BeamPitch = 0.145    # m, mandatory spacing between adjacent beams on the same face
FrameAngleTolDeg = 25.0  # deg, angular slack for splitting observed beam directions into the template's 2 direction classes
ProgressEvery = 1000
PlotEvery = 50  # iters between live redraws of the subset plot while RMDU is running

os.makedirs(SubsetsDir, exist_ok=True)
os.makedirs(GeometryDir, exist_ok=True)


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

with open(os.path.join(SubsetsDir, "subset_manifest.json"), "w") as f:
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


#%%
# ---- Live plot of the current best-quality source's Tau (DTOA) matrix and
# Toa (arrival-time) vector, redrawn as the per-subset source search finds
# progressively better sources. ----
def init_tdoa_plot():
    plt.ion()
    fig, (ax_tau, ax_toa) = plt.subplots(1, 2, figsize=(14, 6))
    return fig, ax_tau, ax_toa


def plot_best_source(ax_tau, ax_toa, mems, Tau, Toa, beam_colors, idx, src_id, quality, good_count, target):
    ax_tau.cla()
    ax_tau.imshow(Tau, cmap="turbo")
    ax_tau.set_title(f"Subset {idx:02d} -- best Tau so far (src {src_id}, Q={quality:.4f})")
    ax_tau.set_xlabel("mem (local idx)")
    ax_tau.set_ylabel("mem (local idx)")

    ax_toa.cla()
    ax_toa.scatter(np.arange(len(Toa)), Toa, c=beam_colors(mems), s=15)
    ax_toa.set_title(f"Subset {idx:02d} -- best Toa so far (good sources {good_count}/{target})")
    ax_toa.set_xlabel("mem (local idx)")
    ax_toa.set_ylabel("arrival time (s)")

    plt.draw()
    plt.pause(0.01)


def process_subset(idx, subset, ax_tau, ax_toa, beam_colors):
    mems = subset["mems"]
    cols = mem_to_col(mems)
    Spark = extract_spark_chunks(cols)
    NbSrcs = Spark.shape[0]

    # Seeded per subset (not just once globally) so each subset still gets its own
    # distinct source order, but the whole run is reproducible under a fixed Seed --
    # this used to draw from the unseeded global numpy random state.
    Order = np.random.default_rng(None if Seed is None else Seed + idx).permutation(NbSrcs)
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
                GoodCount = sum(1 for q in QualAll.values() if q >= QMax * (1 - QualTol))
                if Q > QMax:
                    QMax = Q
                    GoodCount = sum(1 for q in QualAll.values() if q >= QMax * (1 - QualTol))
                    plot_best_source(ax_tau, ax_toa, mems, TauMat, ToaVec, beam_colors,
                                      idx, s, Q, GoodCount, NbGoodTarget)

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
    np.save(os.path.join(SubsetsDir, f"subset_{idx:02d}_TDOA.npy"), Tau)
    np.save(os.path.join(SubsetsDir, f"subset_{idx:02d}_TOA.npy"), Toa)

    fig_all, _, _ = gcu.PlotTau(Tau, Tau.shape[0])
    fig_all.savefig(os.path.join(SubsetsDir, f"subset_{idx:02d}_TDOA.png"))
    plt.close(fig_all)

    return Tau, Toa


#%%
# ---- Cube template: candidate (center, direction) pose for every physical
# beam slot, in the cube's own local coordinates (x,y in [-Half,Half],
# z in [0,CubeSide], bottom face z=0 excluded). Built once at import time. ----
def build_cube_template(cube_side=CubeSide, beams_per_face=BeamsPerFace, beam_pitch=BeamPitch):
    half = cube_side / 2.0
    mid = cube_side / 2.0
    k = np.arange(beams_per_face, dtype=float)
    offsets = (k - (beams_per_face - 1) / 2) * beam_pitch  # centered row of beams, BeamPitch apart

    centers, directions, classes = [], [], []
    for y in offsets:  # top face: beams parallel to x, centered on it, at z=cube_side
        centers.append([0.0, y, cube_side]); directions.append([1.0, 0.0, 0.0]); classes.append("top")
    for sign in (+1.0, -1.0):  # side faces x = +-half: vertical beams, centered in height
        for y in offsets:
            centers.append([sign * half, y, mid]); directions.append([0.0, 0.0, 1.0]); classes.append("side")
    for sign in (+1.0, -1.0):  # side faces y = +-half
        for x in offsets:
            centers.append([x, sign * half, mid]); directions.append([0.0, 0.0, 1.0]); classes.append("side")

    return np.array(centers), np.array(directions), np.array(classes)


TemplateCenters, TemplateDirs, TemplateClass = build_cube_template()
NbTemplateBeams = len(TemplateCenters)


def template_mic_positions_local(slot_idx, n_mics=BeamSize, spacing=BeamSpacing, sign=1.0):
    """8 template mic positions (n,3), local cube coordinates, for the given template slot, spaced at `spacing` and centered on it; `sign` picks which end of the line mem-index 0 (of the beam) sits at."""
    k = np.arange(n_mics, dtype=float)
    offsets = sign * spacing * (k - (n_mics - 1) / 2)
    return TemplateCenters[slot_idx][None, :] + offsets[:, None] * TemplateDirs[slot_idx][None, :]


def fitted_beam_pose(mic_xyz_rows):
    """Centroid (3,) and unit direction (3,, sign arbitrary) of a beam's mics (n,3) via SVD."""
    pts = mic_xyz_rows.T  # (3, n)
    centroid = pts.mean(axis=1)
    centered = pts - centroid[:, None]
    u, _, _ = np.linalg.svd(centered, full_matrices=False)
    return centroid, u[:, 0]


def order_sign(mic_xyz_rows, direction):
    """+1/-1: whether ascending mem order (row order) runs with or against `direction`."""
    pts = mic_xyz_rows.T
    proj = direction @ (pts - pts.mean(axis=1, keepdims=True))
    k = np.arange(pts.shape[1], dtype=float)
    return 1.0 if np.sum((k - k.mean()) * (proj - proj.mean())) >= 0 else -1.0


def cluster_two_directions(canon, angle_tol_deg, min_cluster_size=2):
    """
    Splits n already-canonicalized unit vectors (sign-normalized so exact
    opposites compare as similar -- see the `canon` construction at each call
    site) into two direction clusters via a robust reference pick: whichever
    vector pulls in the largest same-direction group (within angle_tol_deg)
    anchors clusterA; everything else forms clusterB, UNLESS clusterB would
    have fewer than min_cluster_size members, in which case those few are
    folded back into clusterA (meanB is None) instead of being trusted as a
    second direction -- a single dissenting vector's own "cluster" would
    otherwise just be itself, i.e. every such vector trivially agrees with
    its own mean and never reads as an outlier to anything (verified this
    session: a synthetic beam 40 degrees off the other 4 was left untouched
    by enforce_cross_beam_parallelism for exactly this reason before this
    floor was added). A fixed reference (e.g. always the first vector) is
    similarly fragile -- one badly-fit sample landing first would corrupt
    the whole split, which is exactly what corrupted the very first
    subset's cube-bootstrap frame this session before that was made robust.
    Returns (same_mask, meanA, meanB); meanB is None iff clusterB is empty
    or was folded into clusterA.
    """
    sims = np.abs(canon @ canon.T)
    ref = canon[np.argmax((sims >= np.cos(np.radians(angle_tol_deg))).sum(axis=1))]
    same = np.abs(canon @ ref) >= np.cos(np.radians(angle_tol_deg))
    # meanA is always the pure majority mean, computed before any folding below --
    # letting a folded-in dissenter contribute to the very mean it's being pulled
    # toward would dilute the correction into a self-consistent compromise instead
    # of actually converging it back onto the majority (verified this session: a
    # synthetic 40-degree outlier plateaued at ~19 degrees, never closing in on the
    # true 0, until meanA was decoupled from fold membership this way).
    meanA = canon[same].mean(axis=0)
    meanA /= np.linalg.norm(meanA)
    if 0 < (~same).sum() < min_cluster_size:
        same = np.ones_like(same)
    meanB = None
    if (~same).any():
        meanB = canon[~same].mean(axis=0)
        meanB /= np.linalg.norm(meanB)
    return same, meanA, meanB


def discover_cube_frame(beam_poses, angle_tol_deg=FrameAngleTolDeg):
    """
    From a handful of observed (centroid, direction) beam poses, in whatever
    frame the RMDU solver produced, recover the rotation R (3,3, columns
    [X_hat, Y_hat, Z_hat]) mapping the cube template's local axes into that
    frame. The template only has 2 distinct beam directions (vertical for
    the side faces, horizontal for the top face), so a beam's direction
    alone says which class it's in -- no positions needed for that part.
    Z_hat's sign, though, isn't determined by direction alone (an axis has
    no "up"): it's resolved by checking that the horizontal-direction (top
    face) beams sit higher along Z_hat than the vertical-direction (side
    face) ones do, matching the template (z=CubeSide vs z=CubeSide/2) -- the
    template's top/bottom asymmetry (no bottom face) makes this the one
    axis sign that actually matters. X_hat's sign is left arbitrary: which
    physical side face ends up "+x" vs "-x" doesn't matter since TDOA-only
    reconstruction is ambiguous under reflection regardless.
    """
    centroids = np.array([c for _, c, _ in beam_poses])
    dirs = np.array([d / np.linalg.norm(d) for _, _, d in beam_poses])
    canon = dirs * np.sign(dirs[np.arange(len(dirs)), np.argmax(np.abs(dirs), axis=1)])[:, None]

    same, meanA, meanB = cluster_two_directions(canon, angle_tol_deg)
    clusterA, clusterB = canon[same], canon[~same]
    centroidsA, centroidsB = centroids[same], centroids[~same]
    if meanB is None:
        # degenerate sample (e.g. only 1-2 beams seen so far): treat as all-vertical
        clusterB, centroidsB, meanB = clusterA[:1], centroidsA[:1], meanA
    # majority cluster = the side faces' shared vertical direction (48 of 60 template beams)
    if len(clusterA) >= len(clusterB):
        Z_hat, X_hat, Z_centroids, X_centroids = meanA, meanB, centroidsA, centroidsB
    else:
        Z_hat, X_hat, Z_centroids, X_centroids = meanB, meanA, centroidsB, centroidsA

    X_hat = X_hat - np.dot(X_hat, Z_hat) * Z_hat
    X_hat /= np.linalg.norm(X_hat)

    if len(X_centroids) > 0 and (X_centroids @ Z_hat).mean() <= (Z_centroids @ Z_hat).mean():
        Z_hat = -Z_hat

    Y_hat = np.cross(Z_hat, X_hat)
    return np.stack([X_hat, Y_hat, Z_hat], axis=1)  # world_vec = R @ local_vec


def classify_direction(direction, R):
    """'top' or 'side': whichever of the frame's X_hat (top) / Z_hat (side) axes `direction` aligns with better."""
    d = direction / np.linalg.norm(direction)
    return "top" if abs(np.dot(d, R[:, 0])) >= abs(np.dot(d, R[:, 2])) else "side"


def match_beams_to_slots(beam_poses, R, origin, used_slots):
    """
    Hungarian-match each (beam_id, centroid, direction) in beam_poses to the
    nearest not-yet-`used_slots` template slot of the matching direction
    class (top beams only to top slots, side beams only to side slots).
    Returns {beam_id: slot_idx}.
    """
    slot_world = origin + TemplateCenters @ R.T
    assignment = {}
    for cls in ("top", "side"):
        items = [(b, c) for b, c, d in beam_poses if classify_direction(d, R) == cls]
        candidates = np.array([i for i in range(NbTemplateBeams) if TemplateClass[i] == cls and i not in used_slots])
        if not items or len(candidates) == 0:
            continue
        cost = np.linalg.norm(
            np.array([c for _, c in items])[:, None, :] - slot_world[candidates][None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            assignment[items[r][0]] = int(candidates[c])
    return assignment


def discover_frame_and_assignment(beam_poses, search_radius=1.2, search_step=0.05):
    """
    One-time (first-subset) bootstrap: find the rigid frame (R, origin)
    mapping the cube template into the RMDU frame, and the slot assignment
    for `beam_poses`. R comes straight from discover_cube_frame (already
    accurate to direction-fit noise). Origin is 3 numbers but only 2 are
    hard to pin down: its Z_hat component has a closed form (side beams
    all share local z=CubeSide/2 in the template regardless of which face
    they're on, so their mean world projection onto Z_hat is unbiased no
    matter how unevenly the observed beams are spread across faces -- top
    beams, if any, would give the same thing via z=CubeSide). The X_hat/
    Y_hat components don't have that luxury: with beams spaced only
    CubeSide/BeamsPerFace apart, a naive centroid-based guess can be biased
    by over a slot's width toward whichever faces happen to be observed,
    and a plain ICP refinement from that guess reliably locks onto the
    wrong neighboring slot and stays there (verified on synthetic data).
    So those two are found by a small local grid search around the
    centroid-based guess, scored by total Hungarian-assignment residual.
    """
    R = discover_cube_frame(beam_poses)
    X_hat, Y_hat, Z_hat = R[:, 0], R[:, 1], R[:, 2]
    centroids = np.array([c for _, c, _ in beam_poses])
    classes = np.array([classify_direction(d, R) for _, _, d in beam_poses])

    mid = CubeSide / 2.0
    side_c, top_c = centroids[classes == "side"], centroids[classes == "top"]
    oz = (side_c @ Z_hat).mean() - mid if len(side_c) else (top_c @ Z_hat).mean() - CubeSide

    origin0 = centroids.mean(axis=0) - R @ TemplateCenters.mean(axis=0)
    ox0, oy0 = origin0 @ X_hat, origin0 @ Y_hat

    grid = np.arange(-search_radius, search_radius + 1e-9, search_step)
    best_cost, best_ox, best_oy, best_assignment = np.inf, ox0, oy0, {}
    for ox in ox0 + grid:
        for oy in oy0 + grid:
            origin = ox * X_hat + oy * Y_hat + oz * Z_hat
            assignment = match_beams_to_slots(beam_poses, R, origin, used_slots=set())
            if not assignment:
                continue
            cost = sum(
                np.linalg.norm(c - (origin + R @ TemplateCenters[assignment[b]]))
                for b, c, _ in beam_poses if b in assignment)
            if cost < best_cost:
                best_cost, best_ox, best_oy, best_assignment = cost, ox, oy, assignment

    origin = best_ox * X_hat + best_oy * Y_hat + oz * Z_hat
    residuals = [np.linalg.norm(c - (origin + R @ TemplateCenters[best_assignment[b]]))
                 for b, c, _ in beam_poses if b in best_assignment]
    n_top, n_side = int((classes == "top").sum()), int((classes == "side").sum())
    print(f"    [cube bootstrap] {len(beam_poses)} beams (top/side split {n_top}/{n_side}), "
          f"{len(best_assignment)} matched, snap residual mean={np.mean(residuals):.4f}m max={np.max(residuals):.4f}m "
          f"(< {BeamPitch / 2:.3f}m/2 is a confident match; larger means the top/side split or slot pick may be unreliable)")
    return R, origin, best_assignment


#%%
# ---- Constraint helpers, applied in place on (ndim, n) point arrays ----
def enforce_cross_beam_parallelism(Xt, beam_cols, angle_tol_deg=BeamDirTolDeg):
    """
    All beams in a subset share only 2 possible physical directions
    (vertical for side faces, horizontal for top) -- but with only ~12
    sources anchoring a subset's ~150-190 mics through an incomplete
    (bipartite) distance matrix, the classical-MDS "horseshoe" effect can
    smoothly bend fitted beam directions across the array instead of
    resolving to those 2 discrete directions, even from clean, well-
    conditioned TOA data (confirmed this session: check_beam_geometry
    caught beams tens of degrees off, and their raw TOA data was fine --
    several beams' own fitted directions formed a smooth gradient instead
    of a tight 2-cluster split). Mirrors enforce_beam_alignment's soft dead
    zone: each beam's own current fitted direction is clustered against
    every other beam's (see cluster_two_directions), and if it's more than
    angle_tol_deg off its cluster's mean direction, it's rotated -- about
    its own centroid, preserving each mic's along-beam position -- only
    enough to land exactly on that boundary, not fully onto the mean, so
    the solver keeps the freedom to settle on whichever of the 2 directions
    the data actually supports. Run before enforce_beam_alignment so beam
    straightness/pitch gets enforced against the corrected direction.
    """
    dirs, centroids, cols_list = [], [], []
    for cols in beam_cols:
        if len(cols) < 3:
            continue
        pts = Xt[:, cols]
        centroid = pts.mean(axis=1, keepdims=True)
        centered = pts - centroid
        try:
            u, _, _ = la.svd(centered, full_matrices=False)
        except la.LinAlgError:
            continue
        dirs.append(u[:, 0])
        centroids.append(centroid)
        cols_list.append(cols)
    if len(dirs) < 2:
        return

    dirs = np.array(dirs)
    canon = dirs * np.sign(dirs[np.arange(len(dirs)), np.argmax(np.abs(dirs), axis=1)])[:, None]
    same, meanA, meanB = cluster_two_directions(canon, angle_tol_deg)

    for i, cols in enumerate(cols_list):
        target = meanA if same[i] else meanB
        if target is None:
            continue
        d = dirs[i]
        t = target if np.dot(d, target) >= 0 else -target
        theta = np.degrees(np.arccos(np.clip(np.dot(d, t), -1.0, 1.0)))
        if theta <= angle_tol_deg or theta < 1e-6:
            continue

        frac = (theta - angle_tol_deg) / theta
        theta_rad = np.radians(theta)
        new_dir = (np.sin((1 - frac) * theta_rad) * d + np.sin(frac * theta_rad) * t) / np.sin(theta_rad)
        new_dir /= np.linalg.norm(new_dir)

        centroid = centroids[i]
        pts = Xt[:, cols]
        proj = (pts - centroid).T @ d  # (n,), signed distance along the beam's own current direction
        Xt[:, cols] = centroid + np.outer(new_dir, proj)


def enforce_beam_alignment(Xt, beam_cols, spacing=BeamSpacing, tol=BeamAlignTol):
    """
    Beam shape is mandatory -- mics collinear, in physical mem order, at the
    fixed `spacing` pitch -- but only enforced past a `tol` dead zone: for
    each mic, its ideal position is found on the beam's fitted 3D line (SVD
    principal axis, oriented consistently with the current mem order), and
    the mic is left alone if it's already within `tol` of that ideal spot.
    A mic further off is pulled back to just `tol` away from its ideal
    position (not snapped exactly onto it), so the solver stays free to
    move within the small mounting-tolerance band.
    """
    for cols in beam_cols:
        n = len(cols)
        if n < 3:
            continue
        pts = Xt[:, cols]
        centroid = pts.mean(axis=1, keepdims=True)
        centered = pts - centroid
        try:
            u, _, _ = la.svd(centered, full_matrices=False)
        except la.LinAlgError:
            continue
        direction = u[:, 0:1]
        sign = order_sign(pts.T, direction.ravel())
        k = np.arange(n, dtype=float)
        proj_ideal = sign * spacing * (k - (n - 1) / 2)
        ideal_pts = centroid + direction @ proj_ideal[None, :]

        dev = pts - ideal_pts
        dist = np.linalg.norm(dev, axis=0)
        violate = dist > tol
        if np.any(violate):
            scale = tol / dist[violate]
            pts[:, violate] = ideal_pts[:, violate] + dev[:, violate] * scale
            Xt[:, cols] = pts


def project_max_distance(pts, max_dist, n_passes=3, relax=1.0):
    """
    In place: iteratively pull together any pair of columns in `pts`
    (ndim, n) farther apart than max_dist, correcting only `relax` of the
    excess per pass (default 1.0: fully to the boundary). A full correction
    every iteration can lock into a stable back-and-forth with another hard
    constraint pulling the opposite way (seen with max_dist_mics vs
    max_source_dist: not a bug in either, just two simultaneous hard caps
    that don't have a single fixed point, so under-relaxing lets both
    converge toward a shared compromise instead of oscillating forever).
    """
    n = pts.shape[1]
    if n < 2 or max_dist <= 0:
        return pts
    for _ in range(n_passes):
        diff = pts[:, :, None] - pts[:, None, :]
        dist_sq = np.sum(diff ** 2, axis=0)
        violators = np.argwhere(np.triu(dist_sq, 1) > max_dist ** 2)
        if len(violators) == 0:
            break
        for i, j in violators:
            d = np.sqrt(dist_sq[i, j])
            if d < 1e-9:
                continue
            excess = (d - max_dist) / 2 * relax
            correction = (pts[:, i] - pts[:, j]) * (excess / d)
            pts[:, i] -= correction
            pts[:, j] += correction
    return pts


def project_sources_outside(X, Ns, Nr, margin=0.0, max_dist=MaxSourceDist, max_dist_relax=1.0):
    """
    In place: keep every source in the shell between the mic volume and
    `max_dist` from the mic centroid -- outside the volume, but no farther
    than `max_dist` away. "Outside the volume" primarily uses the true
    convex hull of the mic points: a source inside is moved along the
    outward normal of its nearest violated facet, just past it (by
    `margin`). Falls back to the mic bounding sphere (centroid + max mic
    radius) when the hull can't be built -- too few mics, or a
    near-degenerate/coplanar configuration, which happens often while the
    geometry is still converging. The `max_dist` cap is then applied on top
    regardless of which method produced the "outside" position, correcting
    only `max_dist_relax` of the excess per iteration (see project_max_
    distance's docstring: full correction here can lock into a stable
    back-and-forth against max_dist_mics's own full correction).
    """
    mics = X[:, Ns:Nr]
    ndim = X.shape[0]
    if mics.shape[1] == 0:
        return
    src = X[:, :Ns]
    centroid = mics.mean(axis=1, keepdims=True)

    hull = None
    if mics.shape[1] >= ndim + 1:
        try:
            hull = ConvexHull(mics.T)
        except QhullError:
            hull = None

    if hull is not None:
        # equations rows are [normal..., offset]; normal.p + offset <= 0 for p inside the hull.
        normals = hull.equations[:, :ndim]
        offsets = hull.equations[:, ndim]
        V = normals @ src + offsets[:, None]  # (nfacets, Ns): each source's signed value per facet
        vmax = V.max(axis=0)                   # nearest facet's value per source (<=0 while inside)
        inside = vmax <= margin
        if np.any(inside):
            nearest_facet = np.argmax(V[:, inside], axis=0)
            push = margin - vmax[inside]
            move_dirs = normals[nearest_facet]  # (n_inside, ndim)
            src[:, inside] += (move_dirs * push[:, None]).T
    else:
        radius = np.max(la.norm(mics - centroid, axis=0))
        limit = radius + margin

        vec = src - centroid
        dist = la.norm(vec, axis=0)
        tooClose = dist < limit
        if np.any(tooClose):
            safe_dist = np.where(dist > 1e-9, dist, 1.0)
            dirs = np.where(dist > 1e-9, vec / safe_dist, 0.0)
            zeroMask = dist <= 1e-9
            if np.any(zeroMask):
                # Deterministic fallback direction (was np.random.randn drawing from the
                # unseeded global RNG state -- true nondeterminism injected on every hit,
                # inside the per-iteration constraint loop, that could cascade into a
                # completely different final geometry between otherwise-identical runs).
                rnd = np.arange(1, X.shape[0] * int(zeroMask.sum()) + 1, dtype=float).reshape(X.shape[0], -1)
                rnd /= (la.norm(rnd, axis=0, keepdims=True) + 1e-12)
                dirs[:, zeroMask] = rnd
            src[:, tooClose] = centroid + dirs[:, tooClose] * limit

    if max_dist > 0:
        vec = src - centroid
        dist = la.norm(vec, axis=0)
        tooFar = dist > max_dist
        if np.any(tooFar):
            safe_dist = np.where(dist > 1e-9, dist, 1.0)
            target = centroid + (vec[:, tooFar] / safe_dist[tooFar]) * max_dist
            src[:, tooFar] += max_dist_relax * (target - src[:, tooFar])

    X[:, :Ns] = src


#%%
def run_rmdu(toa_matrix, mems, subset, lambda_param, initial_xyz=None, seed=None,
             max_iter=MaxIter, eps_limit=EpsLimit, temp_celsius=TempCelsius, ndim=Ndim,
             stall_window=StallWindow, stall_tol=StallTol,
             progress_every=ProgressEvery, plot_every=None, plot_callback=None, label=""):
    """
    RMDU solver (same Guttman-transform update as GeoCalibApp's SolverThread)
    for one subset's bipartite source-mic TOA matrix. The only constraint
    applied during the solve is that sources stay outside the mic volume's
    convex hull -- beam alignment/pitch, cross-beam parallelism, and the
    mic-mic/source-mic distance caps were all tried and removed: stacking
    them fought both the data and each other (confirmed this session --
    check git history for enforce_beam_alignment/enforce_cross_beam_
    parallelism/project_max_distance if revisiting this). The mandatory
    beam shape/pitch is instead enforced once, post-hoc, by the template
    snap in merge_subset_into_cube. Stops early either on true convergence
    (eps < eps_limit) or once eps has stopped improving by more than
    stall_tol (relative) over the last stall_window iterations. Returns
    (XYZ (Nr,ndim) sources-then-mics, Ns, final_eps, n_iters).
    """
    C = np.sqrt(1.4 * 287 * (temp_celsius + 273))
    # A (source, mic) pair the TOA extraction flagged as untrustworthy (e.g. a whole
    # beam locked onto the wrong peak for one spark) comes through as NaN; it's
    # excluded from the fit via zero weight rather than dropping the whole source.
    valid = ~np.isnan(toa_matrix)
    D_input = (C * np.nan_to_num(toa_matrix)).T  # (Nm, Ns)
    Nm, Ns = D_input.shape
    Nr = Nm + Ns

    D_full = np.zeros((Nr, Nr))
    D_full[Ns:, :Ns] = D_input
    D_full[:Ns, Ns:] = D_input.T
    W_full = np.zeros((Nr, Nr))
    W_full[Ns:, :Ns] = valid.T
    W_full[:Ns, Ns:] = valid
    Dflat = squareform(D_full)
    Wflat = squareform(W_full)

    X = np.zeros((max_iter, ndim, Nr))
    if initial_xyz is None:
        X[0] = np.random.default_rng(seed).standard_normal((ndim, Nr))
    else:
        X[0] = initial_xyz.T[:ndim]

    Lpinv = compute_Lpinv(Nr, W_full)
    Eps = np.zeros(max_iter)
    t_final = 0

    for t in range(max_iter - 1):
        DDt = pdist(X[t].T)
        WeightedDiff = Wflat * (Dflat - DDt)
        O_val = np.sign(WeightedDiff) * np.maximum(np.abs(WeightedDiff) - lambda_param / 2, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = Wflat * (Dflat - O_val) / DDt
            A11 = np.where((DDt > 1e-9) & (Dflat > O_val), term, 0)
        A1 = squareform(A11)
        L1 = np.diag(A1.sum(1)) - A1
        X[t + 1] = X[t] @ L1 @ Lpinv

        # Only remaining hard constraint: sources must stay outside the mic
        # volume's convex hull (the "cubic hull" -- a spark can't originate from
        # inside the physical frame). Beam alignment/pitch, cross-beam
        # parallelism, the mic-mic max-distance cap, and the source max-distance
        # cap are all removed -- stacking them fought the data and each other
        # (confirmed this session: adding cross-beam parallelism on top of the
        # rest produced the worst result yet), and the post-hoc template snap in
        # merge_subset_into_cube already enforces the mandatory beam shape/pitch
        # on the final merged geometry regardless.
        project_sources_outside(X[t + 1], Ns, Nr, max_dist=0)

        diff_norm = la.norm(X[t + 1] - X[t])
        curr_norm = la.norm(X[t + 1])
        Eps[t] = diff_norm / curr_norm if curr_norm > 1e-12 else 0.0
        t_final = t

        if progress_every and (t + 1) % progress_every == 0:
            print(f"    [{label}] iter {t + 1}/{max_iter}  eps={Eps[t]:.3e}")

        if plot_callback and plot_every and (t + 1) % plot_every == 0:
            plot_callback(X[t + 1].T, Ns, label, t + 1)

        if Eps[t] < eps_limit:
            break
        if t >= stall_window and (t + 1) % stall_window == 0:
            if abs(Eps[t] - Eps[t - stall_window]) < stall_tol * max(Eps[t], 1e-9):
                if progress_every:
                    print(f"    [{label}] stalled at iter {t + 1}/{max_iter}, eps={Eps[t]:.3e} (not improving further)")
                break

    XYZ_final = X[t_final + 1].T
    return XYZ_final, Ns, Eps[t_final], t_final + 1


NbRestarts = 8  # plain unconstrained MDS-style random init is prone to bad local minima on
# real (noisy) data -- verified this session: single-seed step-1 runs landed in visibly
# different final shapes subset to subset, some clearly worse than others. Trying several
# seeds and keeping the one with the lowest TOA-vs-geometry residual (not eps, which is a
# relative-change stopping criterion, not a fit-quality measure) makes this robust.


def beam_residual_scores(toa_matrix, mems, subset, XYZ, Ns, temp_celsius=TempCelsius):
    """
    (worst_mic_score, worst_beam_score, median_score): TOA-fit residual (m)
    between XYZ's pairwise distances and the input toa_matrix, as a flat
    median over every valid (mic, source) pair (median_score), as the worst
    of each beam's own median (worst_beam_score), and as the worst of each
    individual mic's own median (worst_mic_score). A whole beam (8 of a
    subset's ~150-190 mics) can rigidly converge to a wrong position/
    orientation while every other beam fits well (confirmed this session via
    check_beam_geometry catching exactly this), and a single mic within an
    otherwise-good beam can too (confirmed after the cross-beam/alignment
    constraints were removed: check_beam_geometry then caught isolated
    single-mic deviations over 2m inside beams that otherwise passed) -- a
    flat median barely moves for either, and a per-beam median can itself
    stay low when only 1 of its 8 mics is bad, so neither alone can tell such
    a restart from a genuinely good one. worst_mic_score is the finest of the
    three and is what restart selection is scored on below.
    """
    valid = ~np.isnan(toa_matrix)
    beam_of_mem = {m: b for b in subset["beams"] for m in range(b * BeamSize, b * BeamSize + BeamSize)}
    mem_beam_id = np.array([beam_of_mem[m] for m in mems])  # (Nm,), beam id per row of toa_matrix.T
    C = np.sqrt(1.4 * 287 * (temp_celsius + 273))
    dist_pred = squareform(pdist(XYZ))[Ns:, :Ns]  # (Nm, Ns), matches toa_matrix.T layout
    resid_full = np.abs(C * toa_matrix.T - dist_pred)  # (Nm, Ns)
    valid_full = valid.T  # (Nm, Ns)
    median_score = np.median(resid_full[valid_full])
    worst_beam_score = max(
        (np.median(resid_full[rows][valid_full[rows]])
         for rows in (np.where(mem_beam_id == b)[0] for b in subset["beams"])
         if valid_full[rows].any()),
        default=np.inf)
    worst_mic_score = max(
        (np.median(resid_full[i][valid_full[i]]) for i in range(resid_full.shape[0]) if valid_full[i].any()),
        default=np.inf)
    return worst_mic_score, worst_beam_score, median_score


def run_rmdu_two_step_best_of(toa_matrix, mems, subset, lambda1, lambda2, n_restarts=NbRestarts, label="", **kwargs):
    """
    Runs the full rough (lambda1) -> refine (lambda2) pipeline from
    n_restarts random inits and keeps whichever restart's *refined* result
    has the lowest worst-mic median TOA-fit residual (see
    beam_residual_scores). Scoring only the rough step and refining just its
    winner (the previous design) risks committing early: lambda2's much
    tighter fit can reorder which rough restart was actually best, so a
    restart that looked mediocre after the rough step can refine into the
    best final result, and vice versa -- every restart earns its own refine
    pass before any of them is judged.
    """
    temp_celsius = kwargs.get("temp_celsius", TempCelsius)
    best = None
    for seed in range(n_restarts):
        XYZ1, Ns1, eps1, it1 = run_rmdu(toa_matrix, mems, subset, lambda1, seed=seed,
                                         label=f"{label}-step1-seed{seed}", **kwargs)
        XYZ2, Ns2, eps2, it2 = run_rmdu(toa_matrix, mems, subset, lambda2, initial_xyz=XYZ1,
                                         label=f"{label}-step2-seed{seed}", **kwargs)
        worst_mic_score, worst_beam_score, score = beam_residual_scores(toa_matrix, mems, subset, XYZ2, Ns2,
                                                                          temp_celsius)
        print(f"    [{label}-seed{seed}] step1 {it1} iters (eps={eps1:.3e}) -> step2 {it2} iters (eps={eps2:.3e}), "
              f"median resid={score:.4f}m worst-beam={worst_beam_score:.4f}m worst-mic={worst_mic_score:.4f}m")
        if best is None or worst_mic_score < best[0]:
            best = (worst_mic_score, worst_beam_score, score, XYZ2, Ns2, eps2, it2)
    worst_mic_score, worst_beam_score, score, XYZ, Ns, eps, nit = best
    print(f"    [{label}] best of {n_restarts} seeds (scored after refine): median TOA-fit residual={score:.4f}m, "
          f"worst-beam median residual={worst_beam_score:.4f}m, worst-mic median residual={worst_mic_score:.4f}m")
    return XYZ, Ns, eps, nit


#%%
def rigid_align(X_target, Y_source):
    """
    R, cx, cy such that (Y_source - cy) @ R + cx best matches X_target (both (n,3)).
    R is allowed to be an improper rotation (reflection, det=-1), not just a proper
    one -- pure-distance TOA/TDOA reconstruction can't distinguish a configuration
    from its mirror image (a full reflection preserves every pairwise distance), so
    each subset's independent RMDU solve has no way to "know" which handedness
    matches the frame the bootstrap subset established. Verified this session: for
    two of five real subsets, allowing the reflection cut the common-beam alignment
    RMSE roughly in half (0.89->0.53m, 0.88->0.56m) versus forcing a proper rotation.
    """
    cx = X_target.mean(axis=0)
    cy = Y_source.mean(axis=0)
    Xc = X_target - cx
    Yc = Y_source - cy
    U, _, Vt = np.linalg.svd(Yc.T @ Xc)
    R = U @ Vt
    return R, cx, cy


def new_merge_state():
    return {"merged_xyz": {}, "raw_xyz": {}, "R": None, "origin": None, "slot_of_beam": {}, "used_slots": set()}


def to_canonical_frame(pts, state):
    """
    Rotates/translates points out of the arbitrary frame the RMDU solver
    happened to converge to and into the cube template's own local frame
    (world = origin + R @ local, so local = (world - origin) @ R) -- i.e.
    the frame where the top face is horizontal at z=CubeSide and the side
    faces are vertical (constant x or y), instead of whatever tilt the
    first-processed subset's bootstrap happened to land on.
    """
    return (pts - state["origin"]) @ state["R"]


def merge_subset_into_cube(state, subset, mems, mic_xyz):
    """
    Bring this subset's mic geometry (mic_xyz, rows aligned to `mems`) into
    the single shared cube-template frame and snap each of its beams onto
    its matching template slot -- mandating the linear alignment, pitch and
    physical placement, so common beams shared by every subset stop
    drifting from being independently (re-)fit each time and re-averaged.

    The very first subset processed bootstraps the frame: discover the
    cube's orientation from the beams' fitted directions (only two exist in
    the template, so direction alone separates top-face from side-face
    beams) and their placement (see discover_frame_and_assignment), which
    fixes (R, origin) for the rest of the run. Every later subset is first
    Procrustes-aligned into that same, now-fixed frame using the mems it
    already shares with what's merged so far (its shared "common beams",
    already snapped onto the template and so an exact alignment target);
    whatever beams it introduces for the first time then get matched to
    their nearest still-unused template slot in that frame.

    A beam is snapped exactly once, the first time any subset includes it
    -- afterwards its mems are read-only in `state["merged_xyz"]`.
    Returns this subset's mics in the shared frame (mic_xyz_global,
    pre-snap, for diagnostics/plotting), rows aligned to `mems`.
    """
    mem_to_local = {m: i for i, m in enumerate(mems)}

    def beam_pose(mem_ids, xyz):
        rows = xyz[[mem_to_local[m] for m in mem_ids]]
        return fitted_beam_pose(rows)

    if state["R"] is None:
        beam_poses = [(b, *beam_pose(range(b * BeamSize, b * BeamSize + BeamSize), mic_xyz)) for b in subset["beams"]]
        R, origin, assignment = discover_frame_and_assignment(beam_poses)
        state["R"], state["origin"] = R, origin
        mic_xyz_global = mic_xyz
    else:
        R, origin = state["R"], state["origin"]
        overlap = [m for m in mems if m in state["merged_xyz"]]
        X_target = np.array([state["merged_xyz"][m] for m in overlap])
        Y_source = np.array([mic_xyz[mem_to_local[m]] for m in overlap])
        Rr, cx, cy = rigid_align(X_target, Y_source)
        mic_xyz_global = (mic_xyz - cy) @ Rr + cx
        rmse = np.sqrt(np.mean(np.sum((mic_xyz_global[[mem_to_local[m] for m in overlap]] - X_target) ** 2, axis=1)))
        print(f"    [merge] aligned via {len(overlap)} shared mics, RMSE={rmse:.4f}m "
              f"(< {BeamAlignTol:.3f}m is expected; larger means this subset disagrees with what's already merged)")

        new_beams = [b for b in subset["beams"] if b not in state["slot_of_beam"]]
        beam_poses = [(b, *beam_pose(range(b * BeamSize, b * BeamSize + BeamSize), mic_xyz_global)) for b in new_beams]
        assignment = match_beams_to_slots(beam_poses, R, origin, state["used_slots"]) if beam_poses else {}
        if beam_poses:
            residuals = [np.linalg.norm(c - (origin + R @ TemplateCenters[assignment[b]]))
                         for b, c, _ in beam_poses if b in assignment]
            print(f"    [merge] {len(new_beams)} new beams, {len(assignment)} matched, "
                  f"snap residual mean={np.mean(residuals):.4f}m max={np.max(residuals):.4f}m "
                  f"(< {BeamPitch / 2:.3f}m is a confident match)")

    state["slot_of_beam"].update(assignment)
    state["used_slots"].update(assignment.values())

    for b in subset["beams"]:
        slot = state["slot_of_beam"].get(b)
        beam_mems = list(range(b * BeamSize, b * BeamSize + BeamSize))
        if slot is None or all(m in state["merged_xyz"] for m in beam_mems):
            continue  # unassignable (shouldn't happen once bootstrapped), or already snapped by an earlier subset
        rows = mic_xyz_global[[mem_to_local[m] for m in beam_mems]]
        sign = order_sign(rows, R @ TemplateDirs[slot])
        snapped = origin + template_mic_positions_local(slot, sign=sign) @ R.T
        for m, p, raw in zip(beam_mems, snapped, rows):
            state["merged_xyz"][m] = p
            state["raw_xyz"][m] = raw  # the actual RMDU solve for this mem, before the exact-template snap --
            # carries whatever small in-plane positioning error the real TOA data implies, which the snap discards.

    return mic_xyz_global


def check_beam_geometry(state, mems, mic_xyz_global, new_beams, idx):
    """
    Validates two known physical facts about the rig against the beams this
    subset just snapped into the shared frame (new_beams): (1) no mic's raw
    (pre-snap) position should be farther than DeviationTolM from its
    template-snapped position, and (2) each beam's own fitted direction
    should agree with its assigned slot's canonical direction to within
    BeamDirTolDeg -- the template has only 2 directions (horizontal for the
    top face, vertical for every side face regardless of which one), so
    this also checks that vertical beams are grossly parallel to every
    other vertical beam even across different (perpendicular) side faces,
    and only top-vs-side beams are perpendicular. Both are print-only
    diagnostics: a violation means the beam's slot assignment or TOA data
    is suspect, not that the rig itself deviates -- neither should happen
    on the real array.
    """
    mem_to_local = {m: i for i, m in enumerate(mems)}
    R = state["R"]
    for b in new_beams:
        slot = state["slot_of_beam"][b]
        beam_mems = list(range(b * BeamSize, b * BeamSize + BeamSize))

        devs = [(m, np.linalg.norm(state["raw_xyz"][m] - state["merged_xyz"][m])) for m in beam_mems]
        bad = [(m, d) for m, d in devs if d > DeviationTolM]
        if bad:
            print(f"    [check] subset {idx:02d} beam {b}: {len(bad)}/{len(beam_mems)} mic(s) exceed the "
                  f"{DeviationTolM:.2f}m deviation bound: " + ", ".join(f"mem {m} ({d:.3f}m)" for m, d in bad))

        rows = mic_xyz_global[[mem_to_local[m] for m in beam_mems]]
        _, fitted_dir = fitted_beam_pose(rows)
        expected_dir = R @ TemplateDirs[slot]
        angle = np.degrees(np.arccos(np.clip(abs(np.dot(fitted_dir, expected_dir)), -1, 1)))
        if angle > BeamDirTolDeg:
            print(f"    [check] subset {idx:02d} beam {b} ({TemplateClass[slot]} face): fitted direction "
                  f"{angle:.1f} deg off its slot's expected direction (> {BeamDirTolDeg:.0f} deg tolerance -- "
                  f"should be parallel to every other {TemplateClass[slot]}-class beam, "
                  f"perpendicular to beams of the other class)")


#%%
def set_equal_aspect(ax, pts):
    if pts.size == 0:
        return
    c = pts.mean(axis=0)
    r = np.max(np.linalg.norm(pts - c, axis=1)) + 1e-6
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    # Equal axis limits alone don't give an equal visual aspect ratio on a 3D
    # axes -- mpl3d's default box shape is non-cubic unless told otherwise.
    ax.set_box_aspect((1, 1, 1))


def make_beam_colors(nb_beams_total, cmap_name="turbo"):
    """
    Color per beam id (a mem's color is fixed by mem//BeamSize) from a continuous
    colormap, normalized over the full [0, nb_beams_total) range so a given beam
    id maps to the same color everywhere (TDOA-stage Toa scatter, local subset
    plots, the merged plot, and the Plotly browser export in
    Calib_Geometry2Browser.py, which uses the same "turbo" colorscale keyed the
    same way) -- stable across every plot in the pipeline.
    """
    cmap = plt.get_cmap(cmap_name)
    norm = max(nb_beams_total - 1, 1)

    def colors_for(mem_ids):
        beam_ids = (np.asarray(mem_ids) // BeamSize) % nb_beams_total
        return cmap(beam_ids / norm)

    return colors_for


def init_geometry_plot():
    plt.ion()
    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    return fig, ax1, ax2


def plot_local(ax1, mems, mic_xyz, src_xyz, title, beam_colors):
    ax1.cla()
    ax1.scatter(mic_xyz[:, 0], mic_xyz[:, 1], mic_xyz[:, 2], c=beam_colors(mems), s=15)
    if src_xyz.size:
        ax1.scatter(src_xyz[:, 0], src_xyz[:, 1], src_xyz[:, 2], c="black", s=30, marker="^", label="sources")
    ax1.set_title(title)
    ax1.legend(loc="upper right")
    all_local = np.vstack([mic_xyz, src_xyz]) if src_xyz.size else mic_xyz
    set_equal_aspect(ax1, all_local)


def plot_merged(ax2, merged_mems, merged_xyz_arr, title, beam_colors):
    ax2.cla()
    ax2.scatter(merged_xyz_arr[:, 0], merged_xyz_arr[:, 1], merged_xyz_arr[:, 2], c=beam_colors(merged_mems), s=15)
    ax2.set_title(title)
    set_equal_aspect(ax2, merged_xyz_arr)


def update_geometry_plot(ax1, ax2, idx, mems, mic_xyz, src_xyz, merged_mems, merged_xyz_arr, nb_subsets, beam_colors):
    plot_local(ax1, mems, mic_xyz, src_xyz, f"Subset {idx:02d} geometry (local frame, color = beam)", beam_colors)
    plot_merged(ax2, merged_mems, merged_xyz_arr,
                f"Merged geometry, measured ({len(merged_xyz_arr)} mics, {idx + 1}/{nb_subsets} subsets)", beam_colors)
    plt.draw()
    plt.pause(0.05)


def make_live_plotter(ax1, mems, beam_colors, idx):
    """Callback for run_rmdu's plot_callback: redraws ax1 with the solver's
    current (not-yet-converged) mic/source estimate, so the subset plot plays
    live while RMDU iterates instead of only refreshing once it's done."""

    def plot_iter(XYZ_iter, Ns, label, iter_num):
        mic_xyz_live = XYZ_iter[Ns:, :]
        src_xyz_live = XYZ_iter[:Ns, :]
        plot_local(ax1, mems, mic_xyz_live, src_xyz_live,
                   f"Subset {idx:02d} -- {label} (iter {iter_num})", beam_colors)
        plt.pause(0.001)

    return plot_iter


#%%
# ---- Stage 1: TDOA/TOA extraction, every subset ----
BeamColors = make_beam_colors(NbTemplateBeams)  # the physical rig's true beam count (60), not just observed ids
FigTdoa, AxTau, AxToa = init_tdoa_plot()

SubsetTau = []
SubsetToa = []
for idx, subset in enumerate(Subsets):
    Tau, Toa = process_subset(idx, subset, AxTau, AxToa, BeamColors)
    SubsetTau.append(Tau)
    SubsetToa.append(Toa)

print("TDOA/TOA extraction done for all subsets:")
for idx in range(len(Subsets)):
    print(f"  subset {idx:02d}: {os.path.join(SubsetsDir, f'subset_{idx:02d}_TOA.npy')}")

#%%
# ---- Trigger-to-emission timing offset: distance = C*TOA assumes the analog
# trigger fires at the exact instant of acoustic emission, but there's a
# real, constant delay (spark-discharge buildup time) common to every
# measurement. Exact value measured directly from the TTL-vs-mems raw signal
# check (see Plot_Etincelle_Signals.py): 5.8ms, common to every mem and every
# subset -- fixed here rather than estimated per run. ----
ToaOffsetSec = 5.8e-3
print(f"TOA offset (measured, TTL-to-mem): {ToaOffsetSec * 1000:.1f}ms")

#%%
# ---- Stage 2: two-step RMDU geometry reconstruction and merge, every subset ----
MergeState = new_merge_state()
FigGeom, Ax1, Ax2 = init_geometry_plot()

for idx, subset in enumerate(Subsets):
    mems = subset["mems"]
    toa = SubsetToa[idx] - ToaOffsetSec  # (NbSrcs, NbMemsSub)

    live_plotter = make_live_plotter(Ax1, mems, BeamColors, idx)

    print(f"Subset {idx:02d}: {NbRestarts} restarts, each rough (lambda={Lambda1}) then refined (lambda={Lambda2}) ...")
    XYZ2, Ns2, eps2, it2 = run_rmdu_two_step_best_of(toa, mems, subset, Lambda1, Lambda2, label=f"{idx:02d}",
                                                      progress_every=0, plot_every=PlotEvery,
                                                      plot_callback=live_plotter)

    mic_xyz = XYZ2[Ns2:, :]
    src_xyz = XYZ2[:Ns2, :]

    before_merged = set(MergeState["merged_xyz"].keys())
    mic_xyz_global = merge_subset_into_cube(MergeState, subset, mems, mic_xyz)
    MergedXYZ = MergeState["merged_xyz"]
    new_beams = sorted({m // BeamSize for m in MergedXYZ.keys() if m not in before_merged})
    check_beam_geometry(MergeState, mems, mic_xyz_global, new_beams, idx)

    np.savez(os.path.join(GeometryDir, f"subset_{idx:02d}_geometry.npz"),
             mems=np.array(mems), mic_xyz_local=mic_xyz, mic_xyz_global=mic_xyz_global,
             src_xyz=src_xyz, Ns=Ns2)

    merged_mems_sorted = sorted(MergedXYZ.keys())
    merged_raw_xyz_arr = to_canonical_frame(np.array([MergeState["raw_xyz"][m] for m in merged_mems_sorted]), MergeState)
    update_geometry_plot(Ax1, Ax2, idx, mems, mic_xyz, src_xyz, merged_mems_sorted, merged_raw_xyz_arr,
                          len(Subsets), BeamColors)

merged_mems_sorted = sorted(MergedXYZ.keys())
merged_xyz_arr = to_canonical_frame(np.array([MergedXYZ[m] for m in merged_mems_sorted]), MergeState)
raw_xyz_arr = to_canonical_frame(np.array([MergeState["raw_xyz"][m] for m in merged_mems_sorted]), MergeState)
deviation = np.linalg.norm(raw_xyz_arr - merged_xyz_arr, axis=1)
np.savez(os.path.join(GeometryDir, "merged_geometry.npz"), mems=np.array(merged_mems_sorted),
         xyz=merged_xyz_arr, xyz_raw=raw_xyz_arr)
FigGeom.savefig(os.path.join(GeometryDir, "merged_geometry.png"))

print(f"Done: merged geometry has {len(merged_mems_sorted)} mics -> "
      f"{os.path.join(GeometryDir, 'merged_geometry.npz')}")
print(f"  raw-vs-snapped deviation: mean={deviation.mean():.4f}m median={np.median(deviation):.4f}m "
      f"max={deviation.max():.4f}m")

build_comparison_html(np.array(merged_mems_sorted), merged_xyz_arr, raw_xyz_arr,
                       os.path.join(GeometryDir, "merged_geometry_compare.html"))

plt.ioff()
plt.show()
