#%%
# Builds the complete microphone array geometry from the per-subset TOA
# matrices produced by Calib_Subsets2TDOA_IMP.py (results/subsets/subset_*_TOA.npy).
#
# For each subset:
#   1. Rough geometry: RMDU with lambda=Lambda1, random init, run to convergence.
#   2. Refined geometry: RMDU with lambda=Lambda2, initialized from step 1's
#      result, run to convergence.
#   Both steps enforce the same constraints: a global mic-mic max distance,
#   linear alignment + mandatory fixed-pitch (BeamSpacing) ordered spacing
#   of the mics within each beam, and sources kept outside the mic volume
#   (its true convex hull, mic bounding sphere as fallback).
#
# Subsets are merged one at a time onto a single physical template: a
# CubeSide-m cube (no bottom face) whose faces each carry a BeamsPerFace x
# BeamPitch grid of beams. The first subset bootstraps this template's pose
# from its own beams' fitted directions/positions; every later subset is
# first Procrustes-aligned into that same frame via the mems it shares with
# what's already merged (its "common beams", see subset_manifest.json), then
# has its new beams matched and snapped onto their nearest template slot.
# The merged geometry and the subset's own local geometry are plotted live
# and refreshed after each subset is processed.
import os
import json
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist
from scipy.spatial import ConvexHull
from scipy.optimize import linear_sum_assignment
try:
    from scipy.spatial import QhullError
except ImportError:
    from scipy.spatial.qhull import QhullError

from rcbox.rmds import compute_Lpinv

InDir = "results/subsets"
OutDir = "results/geometry"
BeamSize = 8
BeamSpacing = 0.23   # m, mandatory pitch between consecutive mics of a beam (beam length = (BeamSize-1)*BeamSpacing = 1.61 m)
BeamAlignTol = 0.005  # m, mandatory but allows this much slack off the ideal collinear/regularly-spaced beam shape

Lambda1 = 0.05      # step 1: rough geometry
Lambda2 = 2.0        # step 2: refined geometry
MaxIter = 10000
EpsLimit = 1e-10
TempCelsius = 26
Ndim = 3

MaxDistMics = 3.5    # m, cap on distance between any two mics in a subset
MaxSourceDist = 3.0   # m, sources capped within this distance from the mic centroid
# max_dist_mics and max_source_dist can be in direct tension (a full correction of one
# undoes the other's, iteration after iteration -- verified on real data: without
# relaxation eps settles into a stable ~0.17-0.28 2-cycle instead of shrinking). Only
# these two are relaxed -- the "stay outside the hull" push isn't part of that
# conflict and stays full-strength. (There used to also be a per-beam max-distance
# cap here, but a beam's shape is now mandatory and exact -- BeamSize-1 gaps of
# BeamSpacing, i.e. ~1.61m -- well under any distance cap that would matter, so
# that constraint never fired; removed rather than kept as dead weight.)
ConstraintRelax = 0.15

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

os.makedirs(OutDir, exist_ok=True)


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

    ref = canon[0]
    same = np.abs(canon @ ref) >= np.cos(np.radians(angle_tol_deg))
    clusterA, clusterB = canon[same], canon[~same]
    centroidsA, centroidsB = centroids[same], centroids[~same]
    if len(clusterB) == 0:
        # degenerate sample (e.g. only 1-2 beams seen so far): treat as all-vertical
        clusterB, centroidsB = clusterA[:1], centroidsA[:1]

    meanA = clusterA.mean(axis=0); meanA /= np.linalg.norm(meanA)
    meanB = clusterB.mean(axis=0); meanB /= np.linalg.norm(meanB)
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
                rnd = np.random.randn(X.shape[0], int(zeroMask.sum()))
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
def run_rmdu(toa_matrix, mems, subset, lambda_param, initial_xyz=None,
             max_iter=MaxIter, eps_limit=EpsLimit, temp_celsius=TempCelsius, ndim=Ndim,
             max_dist_mics=MaxDistMics, max_source_dist=MaxSourceDist,
             stall_window=StallWindow, stall_tol=StallTol,
             progress_every=ProgressEvery, label=""):
    """
    RMDU solver (same Guttman-transform update as GeoCalibApp's SolverThread)
    for one subset's bipartite source-mic TOA matrix, with the beam
    alignment/spacing and distance/volume constraints applied every
    iteration. Stops early either on true convergence (eps < eps_limit) or
    once eps has stopped improving by more than stall_tol (relative) over
    the last stall_window iterations -- with these hard constraints active,
    the solver usually settles into a stable plateau well above eps_limit
    rather than reaching it, so without stall detection every call burns
    the full max_iter budget regardless. Returns (XYZ (Nr,ndim)
    sources-then-mics, Ns, final_eps, n_iters).
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

    mem_to_local = {m: i for i, m in enumerate(mems)}
    beam_cols = []
    for b in subset["beams"]:
        beam_mems = list(range(b * BeamSize, b * BeamSize + BeamSize))
        beam_cols.append([Ns + mem_to_local[m] for m in beam_mems])

    X = np.zeros((max_iter, ndim, Nr))
    if initial_xyz is None:
        X[0] = np.random.randn(ndim, Nr)
    else:
        X[0] = initial_xyz.T[:ndim]

    Lpinv = compute_Lpinv(Nr, W_full)
    Eps = np.zeros(max_iter)
    t_final = 0

    for t in range(max_iter - 1):
        # Beam shape constraint (pre-step): collinear, regularly spaced, order preserved.
        enforce_beam_alignment(X[t], beam_cols)

        DDt = pdist(X[t].T)
        WeightedDiff = Wflat * (Dflat - DDt)
        O_val = np.sign(WeightedDiff) * np.maximum(np.abs(WeightedDiff) - lambda_param / 2, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = Wflat * (Dflat - O_val) / DDt
            A11 = np.where((DDt > 1e-9) & (Dflat > O_val), term, 0)
        A1 = squareform(A11)
        L1 = np.diag(A1.sum(1)) - A1
        X[t + 1] = X[t] @ L1 @ Lpinv

        # Distance constraints (post-step), applied to mics only. max_dist_mics is
        # relaxed (see ConstraintRelax) since it fights max_source_dist below.
        mics = X[t + 1, :, Ns:Nr]
        project_max_distance(mics, max_dist_mics, relax=ConstraintRelax)
        X[t + 1, :, Ns:Nr] = mics

        # Sources must stay outside the (now constrained) mic volume, and within max_source_dist of it.
        project_sources_outside(X[t + 1], Ns, Nr, max_dist=max_source_dist, max_dist_relax=ConstraintRelax)

        diff_norm = la.norm(X[t + 1] - X[t])
        curr_norm = la.norm(X[t + 1])
        Eps[t] = diff_norm / curr_norm if curr_norm > 1e-12 else 0.0
        t_final = t

        if progress_every and (t + 1) % progress_every == 0:
            print(f"    [{label}] iter {t + 1}/{max_iter}  eps={Eps[t]:.3e}")

        if Eps[t] < eps_limit:
            break
        if t >= stall_window and (t + 1) % stall_window == 0:
            if abs(Eps[t] - Eps[t - stall_window]) < stall_tol * max(Eps[t], 1e-9):
                if progress_every:
                    print(f"    [{label}] stalled at iter {t + 1}/{max_iter}, eps={Eps[t]:.3e} (not improving further)")
                break

    XYZ_final = X[t_final + 1].T
    return XYZ_final, Ns, Eps[t_final], t_final + 1


#%%
def rigid_align(X_target, Y_source):
    """R, cx, cy such that (Y_source - cy) @ R + cx best matches X_target (rotation + translation only, both (n,3))."""
    cx = X_target.mean(axis=0)
    cy = Y_source.mean(axis=0)
    Xc = X_target - cx
    Yc = Y_source - cy
    U, _, Vt = np.linalg.svd(Yc.T @ Xc)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    return R, cx, cy


def new_merge_state():
    return {"merged_xyz": {}, "R": None, "origin": None, "slot_of_beam": {}, "used_slots": set()}


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
        for m, p in zip(beam_mems, snapped):
            state["merged_xyz"][m] = p

    return mic_xyz_global


#%%
def set_equal_aspect(ax, pts):
    if pts.size == 0:
        return
    c = pts.mean(axis=0)
    r = np.max(np.linalg.norm(pts - c, axis=1)) + 1e-6
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)


def make_beam_colors(nb_beams_total):
    """Distinct color per beam id, stable across all plots (a mem's color is fixed by mem//BeamSize)."""
    cmap = plt.get_cmap("gist_rainbow")

    def colors_for(mem_ids):
        beam_ids = (np.asarray(mem_ids) // BeamSize) % nb_beams_total
        return cmap(beam_ids / max(nb_beams_total - 1, 1))

    return colors_for


def init_plot():
    plt.ion()
    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    return fig, ax1, ax2


def update_plot(ax1, ax2, idx, mems, mic_xyz, src_xyz, merged_mems, merged_xyz_arr, nb_subsets, beam_colors):
    ax1.cla()
    ax1.scatter(mic_xyz[:, 0], mic_xyz[:, 1], mic_xyz[:, 2], c=beam_colors(mems), s=15)
    if src_xyz.size:
        ax1.scatter(src_xyz[:, 0], src_xyz[:, 1], src_xyz[:, 2], c="black", s=30, marker="^", label="sources")
    ax1.set_title(f"Subset {idx:02d} geometry (local frame, color = beam)")
    ax1.legend(loc="upper right")
    all_local = np.vstack([mic_xyz, src_xyz]) if src_xyz.size else mic_xyz
    set_equal_aspect(ax1, all_local)

    ax2.cla()
    ax2.scatter(merged_xyz_arr[:, 0], merged_xyz_arr[:, 1], merged_xyz_arr[:, 2], c=beam_colors(merged_mems), s=15)
    ax2.set_title(f"Merged geometry ({len(merged_xyz_arr)} mics, {idx + 1}/{nb_subsets} subsets)")
    set_equal_aspect(ax2, merged_xyz_arr)

    plt.draw()
    plt.pause(0.05)


#%%
# ---- Run the two-step RMDU per subset and merge progressively ----
with open(os.path.join(InDir, "subset_manifest.json")) as f:
    Manifest = json.load(f)
Subsets = Manifest["subsets"]
NbBeamsTotal = max(b for subset in Subsets for b in subset["beams"]) + 1
BeamColors = make_beam_colors(NbBeamsTotal)

MergeState = new_merge_state()
Fig, Ax1, Ax2 = init_plot()

for idx, subset in enumerate(Subsets):
    mems = subset["mems"]
    toa = np.load(os.path.join(InDir, f"subset_{idx:02d}_TOA.npy"))  # (NbSrcs, NbMemsSub)

    print(f"Subset {idx:02d}: step 1 (rough, lambda={Lambda1}) ...")
    XYZ1, Ns1, eps1, it1 = run_rmdu(toa, mems, subset, Lambda1, initial_xyz=None, label=f"{idx:02d}-step1")
    print(f"  step 1 converged in {it1} iters, eps={eps1:.3e}")

    print(f"Subset {idx:02d}: step 2 (refine, lambda={Lambda2}) ...")
    XYZ2, Ns2, eps2, it2 = run_rmdu(toa, mems, subset, Lambda2, initial_xyz=XYZ1, label=f"{idx:02d}-step2")
    print(f"  step 2 converged in {it2} iters, eps={eps2:.3e}")

    mic_xyz = XYZ2[Ns2:, :]
    src_xyz = XYZ2[:Ns2, :]

    mic_xyz_global = merge_subset_into_cube(MergeState, subset, mems, mic_xyz)
    MergedXYZ = MergeState["merged_xyz"]

    np.savez(os.path.join(OutDir, f"subset_{idx:02d}_geometry.npz"),
             mems=np.array(mems), mic_xyz_local=mic_xyz, mic_xyz_global=mic_xyz_global,
             src_xyz=src_xyz, Ns=Ns2)

    merged_mems_sorted = sorted(MergedXYZ.keys())
    merged_xyz_arr = np.array([MergedXYZ[m] for m in merged_mems_sorted])
    update_plot(Ax1, Ax2, idx, mems, mic_xyz, src_xyz, merged_mems_sorted, merged_xyz_arr, len(Subsets), BeamColors)

merged_mems_sorted = sorted(MergedXYZ.keys())
merged_xyz_arr = np.array([MergedXYZ[m] for m in merged_mems_sorted])
np.savez(os.path.join(OutDir, "merged_geometry.npz"), mems=np.array(merged_mems_sorted), xyz=merged_xyz_arr)
Fig.savefig(os.path.join(OutDir, "merged_geometry.png"))

print(f"Done: merged geometry has {len(merged_mems_sorted)} mics -> "
      f"{os.path.join(OutDir, 'merged_geometry.npz')}")

plt.ioff()
plt.show()
