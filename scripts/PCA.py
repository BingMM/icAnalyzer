#%% Import 

import numpy as np
import glob
import matplotlib.pyplot as plt
from polplot import pp
from tqdm import tqdm
from icreader import ConductanceImage
from sklearn.decomposition import PCA

#%% Paths

p_in = '/home/bing/Dropbox/work/data/conductance/'

base = '/home/bing/Dropbox/work/code/repos/icAnalyzer/'

#%% Fetch orbits available in all nc files

# Fetch all orbits
o = [int(o[-7:-3]) for o in sorted(glob.glob(p_in + '*.nc'))]

#%% Load all data

H, P, t = [], [], []
for orbit in tqdm(o, total=len(o)):
    filename = p_in + f'or_{str(orbit).zfill(4)}.nc'
    cI = ConductanceImage(filename)
    
    H.append(cI.H)
    P.append(cI.P)
    t.append(cI.time)

H = np.concatenate(H)
P = np.concatenate(P)
t = np.concatenate(t)

lat = cI.grid.lat
lt = (cI.grid.lon/15)%24
shape = cI.grid.shape

#%% PCA

# Assuming `data` is (N, 36, 36)
H_flat = H.reshape((H.shape[0], -1))  # (N, 1296)
H_flat[np.isnan(H_flat)] = 0 # Imputing
pcah = PCA(n_components=12)
hc = pcah.fit_transform(H_flat)

P_flat = P.reshape((P.shape[0], -1))  # (N, 1296)
P_flat[np.isnan(P_flat)] = 0 # Imputing
pcap = PCA(n_components=12)
pc = pcap.fit_transform(P_flat)

#%% Plot PCA

plt.ioff()
fig, axs = plt.subplots(2, 5, figsize=(20, 8))
plt.subplots_adjust(hspace=0, wspace=0)
for i, ax in enumerate(axs[:1, :].flatten()):
    pax = pp(ax)
    pax.plotimg(lat, lt, pcah.components_[i].reshape(shape))
    ax.axis('off')
    ax.text(.5, 1, f'PC {i+1}: {np.round(pcah.explained_variance_ratio_[i]*100,1)}\%', ha='center', va='center', fontsize=16, transform=ax.transAxes)
for i, ax in enumerate(axs[1:, :].flatten()):
    pax = pp(ax)
    pax.plotimg(lat, lt, pcap.components_[i].reshape(shape))
    ax.axis('off')    
axs[0,0].text(0, .5, 'Hall', ha='center', va='center', fontsize=20, transform=axs[0,0].transAxes, rotation='vertical')
axs[1,0].text(0, .5, 'Pedersen', ha='center', va='center', fontsize=20, transform=axs[1,0].transAxes, rotation='vertical')
for ax in axs[1, :]:
    ax.text(.5, 0, 'MLT 00', ha='center', va='center', fontsize=12, transform=ax.transAxes)

plt.savefig('/home/bing/Dropbox/work/code/repos/icAnalyzer/figures/naive_PCA_sl.png', bbox_inches='tight')
plt.close('all')
plt.ion()
