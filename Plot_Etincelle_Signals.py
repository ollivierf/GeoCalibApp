#%%
# Raw-signal check for a single-beam test recording (data/Test_Etincelle_v1.dat,
# 1 counter + 8 mems + 1 TTL trigger channel, per its .log): all 8 mem signals
# and the TTL trigger overlaid on one plot (TTL on its own twin y-axis, its raw
# range being ~1000x the mems'), zoomed to a single spark window so the mem
# waveforms are actually visible instead of dwarfed by the TTL swing.
import os
import numpy as np
import matplotlib.pyplot as plt
from DATParser import DATParser

DatFile = "data/Test_Etincelle_v1.dat"
LogFile = "data/Test_Etincelle_v1.log"
OutFile = "results/Test_Etincelle_v1_signals.png"

SparkIdx = 0       # which detected TTL rising edge to show (0 = first spark)
StartMs = 0.358    # window bounds, ms since the TTL rising edge
EndMs = 0.420

parser = DATParser(DatFile, LogFile)
mics = parser.extract_microphone_signals()   # (Nb_ech, 8)
ttl = parser.extract_analog_signals()[:, 0]  # (Nb_ech,)
meta = parser.get_metadata()
Fe = meta["freq"]

# Same rising-edge detection as the main pipeline's spark detector.
Above = ttl > (ttl.max().astype(np.float64) + ttl.min().astype(np.float64)) / 2
Edges = np.where(Above[1:] & ~Above[:-1])[0] + 1
print(f"{len(Edges)} spark(s) detected; showing spark {SparkIdx} at t={Edges[SparkIdx] / Fe:.4f}s")

edge = Edges[SparkIdx]  # the TTL rising-edge sample itself: ms bounds below are relative to this
i0 = max(edge + int(round(StartMs / 1000 * Fe)), 0)
i1 = min(edge + int(round(EndMs / 1000 * Fe)) + 1, mics.shape[0])
tMs = (np.arange(i0, i1) - edge) / Fe * 1000  # ms, relative to the TTL rising edge
print(f"window [{StartMs}, {EndMs}] ms -> samples [{i0}, {i1}) = {i1 - i0} samples "
      f"(Fe={Fe} Hz -> {1000 / Fe:.3f} ms/sample)")

fig, ax = plt.subplots(figsize=(14, 6))
for m in range(mics.shape[1]):
    ax.plot(tMs, mics[i0:i1, m], linewidth=0.8, marker="o", markersize=3, label=f"mem {m}")
ax.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.5)
ax.set_xlabel("time since TTL rising edge (ms)")
ax.set_ylabel("mem signal (raw ADC counts)")
ax.legend(loc="upper left", ncol=4, fontsize=8)
ax.grid(True, alpha=0.3)

ax2 = ax.twinx()
ax2.plot(tMs, ttl[i0:i1], color="black", linewidth=1.0, marker="o", markersize=3, label="TTL")
ax2.set_ylabel("TTL (raw ADC counts)")
ax2.legend(loc="upper right")

fig.suptitle(f"{os.path.basename(DatFile)} -- spark {SparkIdx}, from TTL rising edge "
             f"(t={edge / Fe:.4f}s), 8 mems + TTL, [{StartMs}, {EndMs}] ms window")
fig.tight_layout()

os.makedirs(os.path.dirname(OutFile), exist_ok=True)
fig.savefig(OutFile, dpi=150)
print(f"Wrote {OutFile}")
plt.show()
