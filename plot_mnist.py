import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns 
sns.set(style='whitegrid')


path = 'results/'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.npy' in file:
            files.append(os.path.join(r, file))

def plot_vals(key):
    plt.figure()
    for f in files:
        (vals, nntype, sigma) = np.load(f, allow_pickle=True)
        if "Nois" in nntype:
            label = f'{nntype}, '+r'$\sigma=$'+f'{sigma:.3f}'
        else:
            label = f'{nntype}'
        if 'acc' in key:
            label = label + f', Best={vals[-1]:.02f}'

        if key in f:
            plt.plot(vals, label=label)
    plt.legend()
    plt.savefig(key+'.pdf', dpi=300)


plot_vals('loss')
plot_vals('acc')