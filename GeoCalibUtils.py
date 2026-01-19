import numpy as np
import matplotlib.pyplot as plt

def procrustes(X, Y):
    centroidX = X.mean(axis=0)
    centroidY = Y.mean(axis=0)
    X_centered = X - centroidX
    Y_centered = Y - centroidY

    cov_matrix = np.dot(Y_centered.T, X_centered)

    U, _, Vt = np.linalg.svd(cov_matrix)

    R = np.dot(U, Vt)

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(U, Vt)

    Y_rotated = np.dot(Y_centered, R)
    Y_rotated += centroidX

    return R, Y_rotated

def PlotTau(Tau, NbSrcs) :         
    fig = plt.figure(figsize=(15, 15))  # Create the figure
    axes = []
    TauMaps = []
    for n in range(NbSrcs) :
        ax = fig.add_subplot(3, (NbSrcs+1)//3, n+1)
        axes.append(ax)
        TauMaps.append(ax.pcolormesh(Tau[n,:,:], cmap ='jet'))
        ax.set_aspect('equal')
        ax.set_title(f'Source {n}')
    fig.tight_layout()
    fig.show()
    return fig, axes, TauMaps

def Plot_XYZ( XYZ, NumFig=200, el = 45, az = 45): 
    fig = plt.figure(NumFig)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*XYZ.T)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')  
    ax.set_zlabel('Z')
    ax.set_aspect('equal')
    ax.view_init(elev=el, azim=az)
    
    
def Iters2PltAnimation(XYZ_Iters, step =1):
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    # Matplotlib Animation
    L = np.max(np.abs(XYZ_Iters))
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, facecolor='black')
    ax.set_facecolor('black')
    ax.grid(False)
    ax.set_box_aspect([1, 1, 1])  # Set equal aspect ratio for 3D plots
    ax.set_xlim([-L, L])
    ax.set_ylim([-L, L])
    ax.set_zlim([-L, L])
    ax.set_xlabel('X', color='white')
    ax.set_ylabel('Y', color='white')
    ax.set_zlabel('Z', color='white')
    ax.tick_params(colors='white')

    XYZ_scatter = ax.scatter([], [], [], c='b', alpha=0.8)
    ax.legend(loc='upper left', facecolor='black', edgecolor='white', labelcolor='white')

    def update(frame):
        XYZ_scatter._offsets3d = (XYZ_Iters[frame,:,0], XYZ_Iters[frame,:,1], XYZ_Iters[frame,:,2])
        return XYZ_scatter

    ani = FuncAnimation(fig, update, frames = XYZ_Iters.shape[0], interval=50, blit=False)
    writer = FFMpegWriter(fps=20, metadata = dict(artist='Me'), bitrate=1800, extra_args=['-loglevel', 'error'])

    ani.save('mic_src_animation.avi', writer=writer)
    plt.close(fig)

def Iters2GOAnimation(XYZ_Iters, step =1):
    import plotly.graph_objects as go
    import plotly as ply
   
    # Plotly Animation
    frames = []
    L = np.max(np.abs(XYZ_Iters))
    axis_range = [-L, L]  # Set the range symmetrically around 0

    for i in range(0, XYZ_Iters.shape[0],step):
        frames.append(go.Frame(data=
            go.Scatter3d(x=XYZ_Iters[i, :, 0], y=XYZ_Iters[i, :, 1], z=XYZ_Iters[i, :, 2], 
                         mode='markers', marker=dict(color='cyan', size=5))))
    layout = go.Layout(
        scene=dict(
            xaxis=dict(backgroundcolor="black", gridcolor="black", showbackground=True, zerolinecolor="black", range = axis_range),
            yaxis=dict(backgroundcolor="black", gridcolor="black", showbackground=True, zerolinecolor="black", range = axis_range),
            zaxis=dict(backgroundcolor="black", gridcolor="black", showbackground=True, zerolinecolor="black", range = axis_range),
        ),
        paper_bgcolor="black",
        font=dict(color="white"),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)]),
                    dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False))])
                ]
            )
        ]
    )

    fig = go.Figure(
        data=[
            go.Scatter3d(x=[], y=[], z=[], mode='markers', marker=dict(color='cyan', size=5), name='Mic'),
            go.Scatter3d(x=[], y=[], z=[], mode='markers', marker=dict(color='magenta', size=5), name='Src')
        ],
        layout=layout,
        frames=frames
    )
# Function to fit points to a line and adjust spacing
def fit_line_and_adjust_spacing_8(points, spacing):
    # Calculate the centroid of the points
    centroid = points.mean(axis=0)
    # Center the points
    centered_points = points - centroid
    # Perform SVD to find the direction vector
    _, _, Vt = np.linalg.svd(centered_points)
    direction_vector = Vt[0]
    # Project the points onto the line defined by the direction vector
    projected_points = np.dot(centered_points, direction_vector[:, np.newaxis]) * direction_vector
    # Adjust the points to have the specified spacing
    adjusted_points = np.zeros_like(projected_points)
    for i in range(len(points)):
        adjusted_points[i] = (i-3.5) * spacing * direction_vector
    # Translate the points back to the original centroid
    adjusted_points += centroid
    return adjusted_points

def compute_patch_variance(matrix, patch_size):
    h, w = matrix.shape
    variances = []
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = matrix[i:i+patch_size, j:j+patch_size]
            variances.append(np.var(patch))
    return np.mean(variances)

def evalDTOA(num, dtoa, dtoamax):
    dm = dtoa*343
    dmmax = dtoamax*343
    dmAvg = np.mean(np.abs(dm))
    dmMax = np.max(dm)  
    dmStd = np.std(dm)
    dmVar = np.var(dm)
    dmRms = np.sqrt(np.mean(dm**2))
    PatchVar = compute_patch_variance(dm, 8)
    #print(f"{num:2d}  | {100*dmAvg/dmmax:.0f} | {100*dmMax/dmmax:.0f} |  {100*dmStd/dmmax:.0f}| {100*np.sqrt(dmVar)/dmmax:.0f} |{100*np.sqrt(PatchVar)/dmmax:.0f}")
    return dmAvg, dmMax, dmStd, dmVar, PatchVar