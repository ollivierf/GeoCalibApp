#%%
# Interactive browser plot of the merged array geometry produced by
# Calib_Sparks2Geometry.py (results/geometry/merged_geometry.npz): the ideal
# (template-snapped) mic positions ("xyz") drawn with transparency, and the
# raw measured positions ("xyz_raw", pre-snap RMDU solve) drawn opaque, on
# the same set of 3D axes, plus thin lines joining each mic's ideal and
# measured position so its deviation is visible at a glance.
#
# build_comparison_html() is called directly from the end of
# Calib_Sparks2Geometry.py's run; this file's own __main__ block is only for
# regenerating the plot standalone from a previously saved merged_geometry.npz.
import os
import webbrowser
import numpy as np
import plotly.graph_objects as go

BeamSize = 8  # mandatory mics per beam, see Calib_Subsets2Geometry.py
Colorscale = "Turbo"


def build_comparison_html(mems, xyz, xyz_raw, out_file, open_browser=True):
    """mems (n,), xyz/xyz_raw (n,3) ideal (template-snapped) vs measured (raw,
    pre-snap) mic positions, same layout as merged_geometry.npz. Writes an
    interactive Plotly HTML to out_file and, by default, opens it."""
    deviation = np.linalg.norm(xyz_raw - xyz, axis=1)
    beam_ids = mems // BeamSize
    hover = [f"mem {m} (beam {b})<br>deviation={dev:.3f}m" for m, b, dev in zip(mems, beam_ids, deviation)]

    fig = go.Figure()

    # deviation segments, ideal -> measured, one per mic
    seg_x, seg_y, seg_z = [], [], []
    for p_ideal, p_meas in zip(xyz, xyz_raw):
        seg_x += [p_ideal[0], p_meas[0], None]
        seg_y += [p_ideal[1], p_meas[1], None]
        seg_z += [p_ideal[2], p_meas[2], None]
    fig.add_trace(go.Scatter3d(
        x=seg_x, y=seg_y, z=seg_z, mode="lines",
        line=dict(color="rgba(120,120,120,0.35)", width=2),
        name="deviation", hoverinfo="skip", showlegend=False,
    ))

    fig.add_trace(go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
        marker=dict(size=4, color=deviation, colorscale=Colorscale, cmin=0, cmax=deviation.max(), opacity=0.25),
        text=hover, hoverinfo="text", name="ideal (template)",
    ))

    fig.add_trace(go.Scatter3d(
        x=xyz_raw[:, 0], y=xyz_raw[:, 1], z=xyz_raw[:, 2], mode="markers",
        marker=dict(size=4, color=deviation, colorscale=Colorscale, cmin=0, cmax=deviation.max(), opacity=1.0,
                    colorbar=dict(title="deviation (m)")),
        text=hover, hoverinfo="text", name="measured (raw)",
    ))

    fig.update_layout(
        title=f"Ideal vs measured geometry -- {len(mems)} mics, "
              f"deviation mean={deviation.mean():.3f}m median={np.median(deviation):.3f}m max={deviation.max():.3f}m",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
        legend=dict(itemsizing="constant"),
    )

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    fig.write_html(out_file, include_plotlyjs="cdn")
    print(f"Wrote {out_file}")
    if open_browser:
        webbrowser.open(os.path.abspath(out_file))


if __name__ == "__main__":
    InFile = "results/geometry/merged_geometry.npz"
    OutFile = "results/geometry/merged_geometry_compare.html"
    d = np.load(InFile)
    build_comparison_html(d["mems"], d["xyz"], d["xyz_raw"], OutFile)
