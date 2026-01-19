#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 18:22:45 2022

@author: francois
"""
#%%
import sys, os


import numpy as np
import scipy.io as io
import warnings

import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import scipy.signal as sig
import os
import rcbox
print(rcbox.__file__)
from rcbox.rmds import RMDU
from rcbox.rtdoa import ROno, denoise_tdoa_soft

plt.close('all')
warnings.filterwarnings("ignore")
#Vitesse du son au moment de la mesure, dependant de la temperature:
Tc = 26 
C = np.sqrt( 1.4 * 287 *(Tc + 273) )  
Path = './'

Xo = np.load('XYZf.npy')
#Xo = None
FileOut = 'XYZff.npy'
toaham = np.load(Path+'toaUp_hammer.npy').T
toavio = np.load(Path+'toaUp_violon.npy').T
toavio[99,9]  = toavio[98,9]
toavio[101,9] = toavio[100,9]

toa = np.concatenate((toaham, toavio), axis = 1)
#toa = toaham
toa = toavio

plt.pcolormesh(toa, cmap = 'jet')
plt.colorbar()
plt.show()

Nm, Ns = toa.shape
Xos = Xo[:Ns,:]
Xom = Xo[Ns+30:,:]
Xo = np.concatenate((Xos, Xom), axis = 0)
################################################################################"    
# %% find geometry from TOA matrix

rtoa_solver = RMDU(C*toa )
rtoa_solver.Run(lbda = 0.050, Xinit = Xo, itmax = 10000, EpsLim = 1e-10, verbose=1)

XYZ_iters = rtoa_solver.X.copy()
NbIter = XYZ_iters.shape[0]
XYZ_last = rtoa_solver.X[-1, ...].copy()
XYZm_last = XYZ_last[Ns:,:]
XYZs_last = XYZ_last[:Ns,:]

L = 1.25
#%% Positions absolues de 16 micros de reference d'espace 3D
## le 1er d'un faisceau sur 2
## pour alignement 
nr = np.arange(3,256,16)
XYZref = np.array([
[2840,	18430,	1470],
[2840,	13160,	1540],
[2860,	7920,	1580],
[2840,	2650,	1600],
[5320,	7900,	1570],
[5330,	15790,	1540],
[7730,	19180,	1530],
[7750,	16560,	1560],
[7710,	11290,	1530],
[7730,	6040,	1610],
[10140,	2630,	1510],
[10120,	13150,	1480],
[10120,	18400,	1550],
[12620,	18420,	1590],
[12610,	13150,	1530],
[12640,	7900,	1540]]
)*1e-4
# Ensemble de réference réduit à 4 micros
Select =  range(0, 16, 4)
nr = nr[Select]
XYZref = XYZref[Select, :]
# recentrage de l'ensemble de reférence
cr = XYZref.mean(axis = 0)
XYZref = XYZref - cr

# recentrage de l'ensemble de mesure correspondant
sub = XYZm_last[nr,:]
cx = sub.mean(axis = 0)
sub_c = sub - cx
# calcul de la matrice de rotation
cov = sub_c.T @ XYZref
u, s, v = np.linalg.svd(cov)
e = np.eye(3)
e[-1, -1] = np.sign(s.prod())
r = (v.T @ e @ u.T).T
# centrage & rotation des itérations de l'ensemble de mesure
cr = cr[None,None,:]
XYZa = (XYZ_iters - cr) @ r
ca = XYZa.mean(axis = 1)[:,None,:]
XYZa = XYZa - ca

XYZmi = XYZa[:,Ns:,:]
XYZsi = XYZa[:,:Ns,:]
XYZm = XYZmi[-1,:,:]
XYZs = XYZsi[-1,:,:]
XYZh = XYZs
XYZh = XYZh - XYZh.mean(axis = 0)
XYZ = np.concatenate((XYZs, XYZm), axis = 0)
#%%
fig = plt.figure()
ax2 = fig.add_subplot(111, projection='3d')
#ax2.scatter(*XYZref.T, marker = 'v',s=100, c='r',label='Ref')
NumF = np.arange(Nm)//8
ax2.scatter(*XYZm.T, c =np.arange(Nm), marker='o',edgecolor='k', s=20, cmap='jet',alpha = 1)
ax2.scatter(*XYZh.T, marker='o',facecolor ='r',edgecolor='k', s=10, alpha=0.5)
#ax2.set_label([str(i) for i in NumF])
ax2.legend()  
L = 2
ax2.set_xlim([-L,L])
ax2.set_ylim([-L,L])
ax2.set_zlim([-L,L])
ax2.set_aspect('equal')

# Set the view angle and distance
ax2.view_init(elev=30, azim=45)  # Set elevation to 30° and azimuth to 45°
ax2.dist = 2  # Set the distance of the view

plt.show(block=True)

np.save(FileOut, XYZ)


Animation = False
if Animation:
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    import plotly.graph_objects as go

    # Matplotlib Animation
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

    mic_scatter = ax.scatter([], [], [], c='b', label='Mic', alpha=0.8)
    src_scatter = ax.scatter([], [], [], c='r', label='Src', alpha=0.8)
    ax.legend(loc='upper left', facecolor='black', edgecolor='white', labelcolor='white')

    def update(frame):
        mic_scatter._offsets3d = (XYZmi[frame,:,0],XYZmi[frame,:,1],XYZmi[frame,:,2])
        src_scatter._offsets3d = (XYZsi[frame,:,0],XYZsi[frame,:,1],XYZsi[frame,:,2])
        return mic_scatter, src_scatter

    ani = FuncAnimation(fig, update, frames=XYZmi.shape[0], interval=50, blit=False)

    writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-loglevel', 'error'])

    ani.save('mic_src_animation.avi', writer=writer)
    plt.close(fig)

    # Plotly Animation
    frames = []
    for i in range(XYZmi.shape[0]):
        frames.append(go.Frame(data=[
            go.Scatter3d(x=XYZmi[i, :, 0], y=XYZmi[i, :, 1], z=XYZmi[i, :, 2], mode='markers', marker=dict(color='cyan', size=5)),
            go.Scatter3d(x=XYZsi[i, :, 0], y=XYZsi[i, :, 1], z=XYZsi[i, :, 2], mode='markers', marker=dict(color='magenta', size=5))
        ]))

    layout = go.Layout(
        scene=dict(
            xaxis=dict(backgroundcolor="black", gridcolor="gray", showbackground=True, zerolinecolor="gray"),
            yaxis=dict(backgroundcolor="black", gridcolor="gray", showbackground=True, zerolinecolor="gray"),
            zaxis=dict(backgroundcolor="black", gridcolor="gray", showbackground=True, zerolinecolor="gray"),
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

    fig.write_html('mic_src_animation.html')

