import sys
import os
import time
import argparse
import multiprocessing as mp

import numpy as np
import h5py
from scipy import optimize

import dragonfly

PREFIX = '/gpfs/exfel/exp/SQS/202102/p002601/scratch/'
NCELLS = 400
NPULSES = 100
ADU_PER_PHOTON = 5.

parser = argparse.ArgumentParser(description='Save hits to emc file')
parser.add_argument('run', help='Run number', type=int)
parser.add_argument('dark_run', help='Dark run number', type=int)
parser.add_argument('-t', '--thresh', help='Hitscore threshold (default: auto)', type=float, default=-1)
args = parser.parse_args()

# Get lit pixels
with h5py.File(PREFIX+'events/r%.4d_events.h5'%args.run, 'r') as f:
    litpix = f['entry_1/litpixels'][:].sum(0)

#sel_litpix = litpix.reshape(-1,NCELLS)[:,::4]
#sel_ind = np.add.outer(np.arange(sel_litpix.shape[0]) * NCELLS, np.arange(0, NCELLS, 4)).ravel()
sel_litpix = litpix.reshape(-1,NCELLS)[:,:NPULSES]
sel_ind = np.add.outer(np.arange(sel_litpix.shape[0]) * NCELLS, np.arange(NPULSES)).ravel()

sel_litpix = sel_litpix.ravel()

# Get hit indices
def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x-x0)**2 / 2 / sigma**2)

if args.thresh == -1:
    hy, hx = np.histogram(sel_litpix, np.arange(0, 65600, 100))
    hcen = 0.5*(hx[1:] + hx[:-1])
    xmax = hy[1:].argmax() + 1 # Ignoring first bin
    popt, pcov = optimize.curve_fit(gaussian, hcen[1:xmax], hy[1:xmax], p0=(hy.max(), xmax, 1000))
    args.thresh = popt[1] + 3*popt[2]
    print('Fitted background Gaussian: %.3f +- %.3f' % (popt[1], popt[2]))

hit_inds = sel_ind[sel_litpix > args.thresh]
print('%d hits using a threshold of %.3f' % (len(hit_inds), args.thresh))

# Write hit indices to events file
with h5py.File(PREFIX+'events/r%.4d_events.h5'%args.run, 'a') as f:
    if 'entry_1/is_hit' in f:
        del f['entry_1/is_hit']
    if 'entry_1/hit_indices' in f:
        del f['entry_1/hit_indices']
    f['entry_1/hit_indices'] = hit_inds

# Get dark offsets
with h5py.File(PREFIX+'dark/r%.4d_dark.h5'%args.dark_run, 'r') as f:
    dark = f['data/mean'][:]
    cells = f['data/cellId'][:]

# Save hits for modules
def worker(module):
    wemc = dragonfly.EMCWriter(PREFIX+'emc/r%.4d_m%.2d.emc' % (args.run, module), 128*512, hdf5=False)
    sys.stdout.flush()
    
    f = h5py.File(PREFIX+'vds/r%.4d.cxi' % args.run, 'r')
    dset = f['entry_1/instrument_1/detector_1/data']

    stime = time.time()

    for i, ind in enumerate(hit_inds):
    #for i, ind in enumerate(hit_inds[:100]):
        cid = f['entry_1/cellId'][ind, module]
        frame = dset[ind, module, 0] - dark[module, np.where(cid==cells)[0][0]]
        phot = np.round(frame/ADU_PER_PHOTON-0.3).astype('i4').ravel()
        wemc.write_frame(phot)
        if module == 0 and (i+1) % 10 == 0:
            sys.stderr.write('\rWritten frame %d/%d (%.3f Hz)' % (i+1, len(hit_inds), (i+1)/(time.time()-stime)))
            sys.stderr.flush()
    if module == 0:
        sys.stderr.write('\n')
        sys.stderr.flush()

    wemc.finish_write()
    f.close()

jobs = [mp.Process(target=worker, args=(m,)) for m in range(16)]
[j.start() for j in jobs]
[j.join() for j in jobs]
sys.stdout.flush()

# Merge modules
print('Merging modules')
det = dragonfly.Detector(PREFIX+'det/det_2601_module.h5')
emods = [dragonfly.EMCReader(PREFIX+'emc/r%.4d_m%.2d.emc' % (args.run, m), det) for m in range(16)]

wemc = dragonfly.EMCWriter(PREFIX+'emc/r%.4d_allq.emc' % args.run, 1024**2, hdf5=False)
stime = time.time()
for i in range(emods[0].num_frames):
    phot = np.array([emods[m].get_frame(i, raw=True) for m in range(16)]).ravel()
    wemc.write_frame(phot)

    if (i+1) % 10 == 0:
        sys.stderr.write('\rWritten frame %d/%d (%.3f Hz)' % (i+1, emods[0].num_frames, (i+1)/(time.time()-stime)))
        sys.stderr.flush()
sys.stderr.write('\n')
sys.stderr.flush()

wemc.finish_write()

# Delete module-wise files
mod_fnames = [PREFIX+'emc/r%.4d_m%.2d.emc' % (args.run, m) for m in range(16)]
[os.remove(fname) for fname in mod_fnames]
print('Deleted module-wise files')

print('DONE')
