#%% Import 

import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from polplot import pp
from tqdm import tqdm
import scipy
from icreader import ConductanceImage
from sklearn.decomposition import PCA

#%% Paths

p_in = '/home/bing/Dropbox/work/data/conductance/'

#%% Fetch orbits available in all nc files

# Fetch all orbits
o = [int(o[-7:-3]) for o in sorted(glob.glob(p_in + '*.nc'))]

#%%

H_list  = []
#P_list  = []
#dH_list = []
#dP_list = []

for orbit in tqdm(o, total=len(o)):
    filename = p_in + f'or_{str(orbit).zfill(4)}.nc'
    cI = ConductanceImage(filename)

    H_list.append(cI.H)
    #P_list.append(cI.P)
    #dH_list.append(cI.dH)
    #dP_list.append(cI.dP)

# Now concatenate once
H  = np.concatenate(H_list, axis=0)
#P  = np.concatenate(P_list, axis=0)
#dH = np.concatenate(dH_list, axis=0)
#P = np.concatenate(dP_list, axis=0)

del H_list#, P_list, dH_list, dP_list
    
#%%

# Assuming `data` is (N, 36, 36)
flat_data = H.reshape((H.shape[0], -1))  # (N, 1296)
valid_idx = ~np.isnan(flat_data).any(axis=1)
pca = PCA(n_components=36*36)
components = pca.fit_transform(flat_data[valid_idx])
#pca.singular_values_

flat_data[np.isnan(flat_data)] = 0

#%%

plt.figure()
plt.plot(pca.singular_values_/np.sum(pca.singular_values_)*100)
plt.plot(np.cumsum(pca.singular_values_/np.sum(pca.singular_values_)*100))


#%%
fig, axs = plt.subplots(6, 9, figsize=(9, 6))
for i, ax in enumerate(axs.flatten()):
    im = ax.imshow(pca.components_[i].reshape(36,36))
    cbar = fig.colorbar(im, ax=ax)    

#%%

mean_map = np.nanmean(H, axis=0)
std_map = np.nanstd(H, axis=0)
skew_map = scipy.stats.skew(H, axis=0, nan_policy='omit')

#%%

fig, axs = plt.subplots(3, 1, figsize=(10,10))
im = axs[0].imshow(mean_map)
axs[0].set_title('mean')
cbar = fig.colorbar(im, ax=axs[0])

im = axs[1].imshow(std_map)
axs[1].set_title('std')
cbar = fig.colorbar(im, ax=axs[1])

im = axs[2].imshow(skew_map)
cbar = fig.colorbar(im, ax=axs[2])
axs[2].set_title('skew')

#%%

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from kneed import KneeLocator

inertias = []
k_range = np.arange(2, 35)
for k in tqdm(k_range, total=k_range.size):
    kmeans = KMeans(n_clusters=k, n_init='auto').fit(flat_data[valid_idx])
    inertias.append(kmeans.inertia_)

knee = KneeLocator(k_range, inertias, curve='convex', direction='decreasing')

plt.figure()
plt.plot(k_range, inertias, 'o-')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (within-cluster sum of squares)")
plt.title(f"Elbow Method for Optimal k: {knee.knee}")
plt.grid()
plt.show()

#%%


N = 11
rows

kmeans = KMeans(n_clusters=11, random_state=0)
labels = kmeans.fit_predict(flat_data[valid_idx])

unique, counts = np.unique(labels, return_counts=True)
for u, c in zip(unique, counts):
    print(f"Cluster {u}: {c} images")


# Cluster centroids: already in flat space
centroids = kmeans.cluster_centers_

fig, axs = plt.subplots(3, 4)
for i, ax in enumerate(axs.flatten()):  # one for each cluster
    img = centroids[i].reshape((36, 36))    
    im = ax.imshow(img, cmap='viridis')
    ax.set_title(f"Cluster {i} Centroid")
    cbar = fig.colorbar(im, ax=ax)
    plt.axis('off')


#%%

import numpy as np
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import BayesianRidge  # or HistGradientBoostingRegressor

def iterative_impute_and_run_pca(flat_data_with_nans, n_components=10, plot=True, image_shape=(36, 36)):
    """
    Uses IterativeImputer to handle NaNs, then runs PCA.

    Parameters:
        flat_data_with_nans: ndarray of shape (N, D) with NaNs
        n_components: number of PCA components
        plot: whether to show plots
        image_shape: shape of original image (default 36x36)

    Returns:
        imputed_data: ndarray (N, D) with imputed values
        components: PCA-transformed data
        pca_model: trained PCA object
    """

    print("🔄 Imputing missing values using IterativeImputer...")
    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)
    imputed_data = imputer.fit_transform(flat_data_with_nans)

    print("🔍 Performing PCA on imputed data...")
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(imputed_data)

    if plot:
        # Explained variance plot
        plt.figure()
        plt.plot(np.arange(1, n_components+1), pca.explained_variance_ratio_, 'o-')
        plt.xlabel("Principal Component")
        plt.ylabel("Explained Variance Ratio")
        plt.title("PCA Variance Explained")
        plt.grid()
        plt.show()

        # Spatial mode visualization
        for i in range(min(n_components, 5)):
            mode = pca.components_[i].reshape(image_shape)
            plt.figure()
            plt.imshow(mode, cmap='RdBu_r')
            plt.title(f"PCA Mode {i+1}")
            plt.colorbar()
            plt.axis('off')
            plt.show()

    return imputed_data, components, pca



#%%

# flat_data_with_nans: shape (450000, 1296), dtype float32/float64, with NaNs
imputed_data, pca_components, pca_model = iterative_impute_and_run_pca(flat_data[:10], n_components=10)



#%% Func

def get_c_scales(cI):
    c_scales = {'wicm': (0, np.round(np.nanmax(cI.wic_avg)+1)),
                'wics': (0, np.round(np.nanmax(cI.wic_std)+1)),
                's12m': (0, np.round(np.nanmax(cI.s12_avg)+1)),
                's12s': (0, np.round(np.nanmax(cI.s12_std)+1)),
                's13m': (0, np.round(np.nanmax(cI.s13_avg)+1)),
                's13s': (0, np.round(np.nanmax(cI.s13_std)+1)),
                'E0':   (0,  25),
                'dE0':  (0, np.round(np.nanmax(cI.dE0)+1)),
                'Fe':   (0, np.round(np.nanmax(cI.Fe)+1)),
                'dFe':  (0, np.round(np.nanmax(cI.dFe)+1)),
                'R':    (0, 150),
                'dR':   (0, np.round(5*np.median(cI.dR[~np.isnan(cI.dR)])+1)),
                'H':    (0, np.round(np.nanmax(cI.H)+1)),
                'dH':   (0, np.round(np.nanmax(cI.dH)+1)),
                'P':    (0, np.round(np.nanmax(cI.P)+1)),
                'dP':   (0, np.round(np.nanmax(cI.dP)+1))
                }
    return c_scales
    
def plot(cI, i, c_scales, lat, lt):    
        
    fig, axs = plt.subplots(3, 6, figsize=(30, 15))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    axes = axs.flatten()[:-2]
    axs[2, 4].set_axis_off()
    axs[2, 5].set_axis_off()
    
    var = [cI.wic_avg[i], cI.wic_std[i], cI.R[i],  cI.dR[i],  cI.H[i], cI.dH[i],
           cI.s13_avg[i], cI.s13_std[i], cI.E0[i], cI.dE0[i], cI.P[i], cI.dP[i],
           cI.s12_avg[i], cI.s12_std[i], cI.Fe[i], cI.dFe[i]]
    
    cs = [c_scales['wicm'], c_scales['wics'], c_scales['R'],  c_scales['dR'],  c_scales['H'], c_scales['dH'],
          c_scales['s13m'], c_scales['s13s'], c_scales['E0'], c_scales['dE0'], c_scales['P'], c_scales['dP'],
          c_scales['s12m'], c_scales['s12s'], c_scales['Fe'], c_scales['dFe']]
    
    tit = ['avg WIC counts', 'std WIC counts', 'WIC*/S13* (R)', 'R std', 'Hall', 'Hall std',
            'avg S13 counts', 'std S13 counts', 'E0', 'E0 std', 'Pedersen', 'Pedersen std',
            'avg S12 counts', 'std S12 counts', 'Fe', 'Fe std']
    
    for j, (ax, var_, cs_, tit_) in enumerate(zip(axes, var, cs, tit)):
        pax = pp(ax)
        if j == 12:
            pax.writeLTlabels(fontsize=16)
            ax.text(.85, .1, '50$^{\circ}$', ha='center', va='center', fontsize=16, transform=ax.transAxes)
        pax.plotimg(lat, lt, var_, crange=cs_)
        ax.set_title(tit_, fontsize=18)
        ax.text(.85, .85, str(int(cs_[-1])), ha='left', va='center', fontsize=16, transform=ax.transAxes)
    
    axs[0,2].text(1.1, 1.2, cI.time[i], ha='center', va='center', fontsize=20, transform=axs[0,2].transAxes)
    
#%% Plot

plt.ioff()
for orbit in tqdm(o, total=len(o)):
    filename = p_in + f'or_{str(orbit).zfill(4)}.nc'
    
    cI = ConductanceImage(filename)
    
    c_scales = get_c_scales(cI)
    
    lat = cI.grid.lat
    lt = (cI.grid.lon/15)%24
    
    p_out_o = p_out + f'or_{str(orbit).zfill(4)}/'
    os.makedirs(p_out_o, exist_ok=True)
    
    for i in range(cI.shape[0]):
        plot(cI, i, c_scales, lat, lt)
        plt.savefig(p_out_o + f'{str(i).zfill(3)}.png', bbox_inches='tight')
        plt.close('all')
