# ********************************************************** #
#    NAME: Blake Cole                                        #
#    ORGN: (self)                                            #
#    FILE: mv_tide_sim.py                                 #
#    DATE: 17 MAR 2025                                       #
# ********************************************************** #
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import geotools as gt

# =============================================================================
# Data Setup
# =============================================================================
# Import land boundary files
print("Loading WH boundary data... ")
wh_shoreline = np.array(gt.read_kml_points('../data/wh_shoreline.kml'))
print("Success.")

print("Loading MV boundary data... ")
mv_shoreline = np.array(gt.read_kml_points('../data/mv_shoreline.kml'))
print("Success.")

# Import tidal current data (.mat -> numpy arrays)
print("Loading tidal current data... ")
data = loadmat('../data/tidal_data_2022_07_30.mat')
u = data['U_store']
v = data['V_store']
t_local_str = data['glocals']
t = np.squeeze(data['thours'])
lat = np.squeeze(data['lat'])
lon = np.squeeze(data['lon'])
waypts = data['waypts']
print("Success.\n")

n_points = np.size(lon) # number of spatial points
n_frames = np.size(t)   # number of timesteps

# Configure Pyplot
plt.style.use('dark_background')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif']
plt.rcParams['figure.dpi'] = 150

# =============================================================================
# Plot / Animate
# =============================================================================
fig, ax = plt.subplots()
ax.plot(wh_shoreline[:,0], wh_shoreline[:,1], color='white', linewidth=1.0, label="WH Shoreline")
ax.plot(mv_shoreline[:,0], mv_shoreline[:,1], color='white', linewidth=1.0, label="MV Shoreline")
q = ax.quiver(lon, lat, u[:,0], v[:,0], color='cyan', scale=25)
qk = ax.quiverkey(q, X=0.9, Y=1.05, U=1, label='1 m/s', labelpos='E', coordinates='axes')

ax.grid(alpha=0.5)
ax.axis('square')
ax.set_xlim(-70.70, -70.58)
ax.set_ylim(41.44, 41.55)
ax.set_xlabel("Longitude [deg]")
ax.set_ylabel("Latitude [deg]")
title_strings = [f"Vineyard Sound Tidal Current\n{t}" for t in t_local_str]
ax.set_title(title_strings[0])  # initial title

def update(frame):
    # Update the quiver data and title
    q.set_UVC(u[:,frame], v[:,frame])
    ax.set_title(title_strings[frame])
    if frame % 10 == 0:
        print(f"Rendering frame {frame}/{n_frames}")
    return [q, ax.title]


# Create the animation with the updated data for each timestep
print("Creating figure animation...")
ani = animation.FuncAnimation(fig, update, frames=np.size(t), interval=100, blit=False)

ani.save("../fig/mv_tides.gif", writer="pillow", fps=10)