import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.linalg
from evaltf import evaltf



def load(fdir):
    At = np.load(fdir + 'At.npy')
    Et = np.load(fdir + 'Et.npy')
    bt = np.load(fdir + 'bt.npy')
    ct = np.load(fdir + 'ct.npy')
    return {'At': At, 'Et': Et, 'bt': bt, 'ct': ct,}


exc_site = 1
molec = 'H2S'
size = 1
scratch_folder = f'mor_ip_no_{molec}_mpi{size}'
freq_str = 'fs_-94.55_-90.55_0.04'
fdir = f"/pscratch/sd/e/erikaye/gfdmrg/{scratch_folder}/{freq_str}/exc{exc_site}/"

fmin, fmax, df = -94.55, -90.55, 0.001
new_freq_str = f'fs_{fmin}_{fmax}_{df}'
freqs = np.arange(fmin, fmax, df)
data = load(fdir)

eta = 0.005
At = data['At']
Et = data['Et']
bt = data['bt']
ct = data['ct']

sigma = -1 / np.pi * np.imag(evaltf(At, Et, bt, ct, freqs + 1.j*eta))


print('save', fdir + f'sigmas_{new_freq_str}'),
np.save(fdir + f'sigmas_h{eta}_{new_freq_str}', sigma)
