import numpy as np
import matplotlib.pyplot as plt
import random
import glob
from tqdm import tqdm
from icreader import ConductanceImage

#%% Paths

p_in = '/home/bing/Dropbox/work/data/conductance/'

#%% Fetch orbits available in all nc files

# Fetch all orbits
o = [int(o[-7:-3]) for o in sorted(glob.glob(p_in + '*.nc'))]

#%%

H, P, dH, dP  = [], [], [], []
for orbit in tqdm(o, total=len(o)):
    filename = p_in + f'or_{str(orbit).zfill(4)}.nc'
    cI = ConductanceImage(filename)
    H.append(cI.H)
    P.append(cI.P)
    dH.append(cI.dH)
    dP.append(cI.dP)

#%%

assert H.shape == P.shape == dH.shape == dP.shape
N = H.shape[0]

seq_size = 5

results = []

# -- Classification loop function --
def classify_images(H, P, dH, dP):
    classified = []
    keep_going = True

    cb, cg = 0, 0
    while keep_going:
        ids = random.randint(0, N - 1)
        Hs, Ps, dHs, dPs = H[ids], P[ids], dH[ids], dP[ids]

        ns = Hs.shape[0]
        idt = random.randint(0, ns - 1)
        if (idt-seq_size) < 0:
            idt = seq_size
        if (idt + seq_size) > ns:
            idt = ns - seq_size
        start, stop = idt - seq_size, idt + seq_size
        
        idt = np.arange(start, stop+1)
        
        
        
        
        


        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        
        im = axs[0,0].imshow(H[idx], cmap='viridis', vmin=0, vmax=40)
        fig.colorbar(im, ax=axs[0,0])
        axs[0,0].set_title(f'Hall - Index {idx}')
        
        im = axs[0,1].imshow(P[idx], cmap='viridis', vmin=0, vmax=10)
        fig.colorbar(im, ax=axs[0,1])
        axs[0,1].set_title(f'Pedersen - Index {idx}')
        
        im = axs[1,0].imshow(dH[idx], cmap='viridis', vmin=0, vmax=65)
        fig.colorbar(im, ax=axs[1,0])
        axs[1,0].set_title(f'Hall std - Index {idx}')
        
        im = axs[1,1].imshow(dP[idx], cmap='viridis', vmin=0, vmax=65)
        fig.colorbar(im, ax=axs[1,1])
        axs[1,1].set_title(f'Pedersen std - Index {idx}')
        
        for ax in axs.flatten():
            ax.axis('off')
        plt.suptitle(f"Press 1 (bad), 9 (good), 5 (save and exit), #bad: {cb}, #good: {cg}")
        plt.tight_layout()

        pressed = []

        def on_key(event):
            if event.key in ['1', '9', '5']:
                pressed.append(event.key)
                plt.close()

        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show(block=True)

        if pressed:
            key = pressed[0]
            if key == '1':
                classified.append((idx, 0))
            elif key == '9':
                classified.append((idx, 1))
            elif key == '5':
                keep_going = False
                print("Stopping and saving...")

        cb = np.sum(np.array(classified)[:, 1] == 0)
        cg = np.sum(np.array(classified)[:, 1] == 1)

    return classified

# -- Run the classification session --
results = classify_images(H, P, dH, dP)

# -- Save the results --
if results:
    indices, labels = zip(*results)
    np.savez('/home/bing/Dropbox/work/temp_storage/classification_results.npz', indices=np.array(indices), labels=np.array(labels))
    print(f"Saved {len(results)} classifications to 'classification_results.npz'")
else:
    print("No classifications made.")

#%%

q = np.load('/home/bing/Dropbox/work/temp_storage/classification_results.npz')
indices = q['indices']
labels = q['labels']


