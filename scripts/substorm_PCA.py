#%% Import 

import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from polplot import pp
from tqdm import tqdm
from icreader import ConductanceImage
from sklearn.decomposition import PCA
from datetime import datetime, timedelta

#%% Paths

p_in = '/home/bing/Dropbox/work/data/conductance/'

base = '/home/bing/Dropbox/work/code/repos/icAnalyzer/'

#%% Load substorm list

# Read the CSV with datetime parsing
sl = pd.read_csv(base + 'data/newell_substorm_list.csv', parse_dates=['Date_UTC'])
slt = np.array([datetime(t.year, t.month, t.day, t.hour, t.minute, t.second) for t in sl['Date_UTC']])

#%% Fetch orbits available in all nc files

# Fetch all orbits
o = [int(o[-7:-3]) for o in sorted(glob.glob(p_in + '*.nc'))]

#%% Load all data

H, P = [], []
sc = 0
for orbit in tqdm(o, total=len(o)):
    filename = p_in + f'or_{str(orbit).zfill(4)}.nc'
    cI = ConductanceImage(filename)

    if np.any((cI.time[0] <= slt) & (slt <= cI.time[-1])):
        slto = slt[np.argmin(abs(slt - cI.time[cI.shape[0]//2]))]
        
        f = (slto <= cI.time + timedelta(minutes=90)) & (cI.time <= slto + timedelta(minutes=120))

        H.append(cI.H[f])
        P.append(cI.P[f])
        
        sc += 1

print(sc)

# Now concatenate once
H  = np.concatenate(H, axis=0)
P  = np.concatenate(P, axis=0)

lat = cI.grid.lat
lt = (cI.grid.lon/15)%24
shape = cI.grid.shape

H_mean = np.nanmean(H, axis=0).reshape(shape)
P_mean = np.nanmean(P, axis=0).reshape(shape)
H_std = np.nanstd(H, axis=0).reshape(shape)
P_std = np.nanstd(P, axis=0).reshape(shape)

#%% Plot mean and std

plt.ioff()
fig, axs = plt.subplots(2, 2, figsize=(15, 15))
plt.subplots_adjust(hspace=0, wspace=0)

for ax, var in zip(axs.flatten(), [H_mean, P_mean, H_std, P_std]):
    pax = pp(ax)
    pax.plotimg(lat,lt,var)
    ax.text(.85, .85, f'max={int(np.max(var))}', ha='center', va='center', fontsize=16, transform=ax.transAxes)

plt.suptitle('Mean and std of all data', y=.95, fontsize=25)

for ax, lab in zip(axs[0, :], ['Hall', 'Pedersen']):
    ax.text(.5, 1, lab, ha='center', va='center', fontsize=20, transform=ax.transAxes)
for ax, lab in zip(axs[:, 0], ['Mean', 'Std']):
    ax.text(0, .5, lab, ha='center', va='center', fontsize=20, transform=ax.transAxes, rotation='vertical')
for ax in axs[-1, :]:
    ax.text(.5, 0, 'MLT=00', ha='center', va='center', fontsize=16, transform=ax.transAxes)

plt.savefig('/home/bing/Dropbox/work/code/repos/icAnalyzer/figures/mean_std_sl.png', bbox_inches='tight')
plt.close('all')
plt.ion()
    
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
