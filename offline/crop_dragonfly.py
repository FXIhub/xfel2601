import sys
import argparse

import numpy as np

import dragonfly

PREFIX = '/gpfs/exfel/exp/SQS/202102/p002601/scratch/'

parser = argparse.ArgumentParser(description='Generate lowq and medq emc files')
parser.add_argument('run', help='Run number', type=int)
args = parser.parse_args()

det = dragonfly.Detector(PREFIX+'det/det_2601_allq01.h5')
emc = dragonfly.EMCReader(PREFIX+'emc/r%.4d_allq.emc'%args.run, det)

wemc = dragonfly.EMCWriter(PREFIX+'emc/r%.4d_lowq.emc'%args.run, 4*128*128, hdf5=False)
for i in range(emc.num_frames):
    phot = emc.get_frame(i, raw=True).reshape(16,128,512)
    wemc.write_frame(phot[[0,7,8,15],:,:128].ravel())
    sys.stderr.write('\r%d/%d'%(i+1, emc.num_frames))
    sys.stderr.flush() # For SLURM
sys.stderr.write('\n')
wemc.finish_write()

wemc = dragonfly.EMCWriter(PREFIX+'emc/r%.4d_medq.emc'%args.run, 8*128*256, hdf5=False)
for i in range(emc.num_frames):
    phot = emc.get_frame(i, raw=True).reshape(16,128,512)
    wemc.write_frame(phot[[0,1,6,7,8,9,14,15],:,:256].ravel())
    sys.stderr.write('\r%d/%d'%(i+1, emc.num_frames))
    sys.stderr.flush() # For SLURM
sys.stderr.write('\n')
wemc.finish_write()
