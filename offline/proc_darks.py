#!/usr/bin/env python

''' Process DSSC darks to get mean, sigma and masks '''

import sys
import os
import time
import glob
import argparse
import multiprocessing as mp
import ctypes
import numpy as np
import h5py

EXPT_PREFIX = '/gpfs/exfel/exp/SQS/202102/p002601/'
DET_NAME = 'SQS_DET_DSSC1M-1'
CHUNK_SIZE = 32

class ProcDarks():
    def __init__(self, run_num, mask_only=False, out_fname=None):
        self.run_num = run_num
        self.mask_only = mask_only

        self._get_cellids()
        
        if out_fname is None:
            self.out_fname = 'r%.4d_dark.h5'%self.run_num
        else:
            self.out_fname = out_fname

    def run(self):
        if self.mask_only:
            self.calculate_masks()
            return

        num_files = len(glob.glob(EXPT_PREFIX + '/raw/r%.4d/*DSSC00*.h5'%self.run_num))
        print(num_files, 'files per module in run', self.run_num)
        sys.stdout.flush()

        num_cells = len(self.cellids)
        marray = mp.Array(ctypes.c_double, num_files*16*num_cells*512*128)
        sarray = mp.Array(ctypes.c_double, num_files*16*num_cells*512*128)
        numarray = mp.Array(ctypes.c_int32, num_files*16*num_cells)

        jobs = [mp.Process(target=self._worker, args=(i, marray, sarray, numarray)) for i in range(16*num_files)]
        [j.start() for j in jobs]
        [j.join() for j in jobs]

        np_marray = np.frombuffer(marray.get_obj(), dtype='f8').reshape(num_files,16,num_cells,128,512)
        np_sarray = np.frombuffer(sarray.get_obj(), dtype='f8').reshape(num_files,16,num_cells,128,512)
        np_numarray = np.frombuffer(numarray.get_obj(), dtype='i4').reshape(num_files,16,num_cells)

        np_marray = (np_marray*np_numarray[:,:,:,np.newaxis,np.newaxis]).sum(0) / np_numarray.sum(0)[:,:,np.newaxis,np.newaxis]
        np_sarray = (np_sarray*np_numarray[:,:,:,np.newaxis,np.newaxis]).sum(0) / np_numarray.sum(0)[:,:,np.newaxis,np.newaxis]
        np_numarray = np_numarray.sum(0)

        with h5py.File(self.out_fname, 'w') as f:
            f['data/mean'] = np_marray
            f['data/sigma'] = np_sarray
            f['data/num'] = np_numarray
            f['data/cellId'] = self.cellids
    
    def calculate_masks(self):
        f = h5py.File(self.out_fname, 'r+')
        mean = f['data/mean'][:]
        sigma = f['data/sigma'][:]
        badpix = np.ones(mean.shape, dtype='u1')

        for c in range(mean.shape[1]):
            cmean = mean[:,c]
            # Median and StDev for each cell
            medmean = np.median(cmean)

            sel = (np.abs(cmean-medmean) <= 1000)
            medmean = np.median(cmean[sel])
            stdmean = np.std(cmean[sel])

            sel = (np.abs(cmean-medmean) < 3*stdmean)
            medmean = np.median(cmean[sel])
            stdmean = np.std(cmean[sel])
            
            sel = (np.abs(cmean-medmean) < 4*stdmean)
            badpix.transpose(1,0,2,3)[c,sel] = 0

        if 'data/badpix' in f:
            del f['data/badpix']
        f['data/badpix'] = badpix
        f.close()

    def _get_cellids(self):
        fname = EXPT_PREFIX+'/raw/r%.4d/RAW-R%.4d-DSSC00-S00000.h5'%(self.run_num, self.run_num)
        with h5py.File(fname, 'r') as fptr:
            cids = fptr['INSTRUMENT/'+DET_NAME+'/DET/0CH0:xtdf/image/cellId'][:1600,0]
        self.cellids = np.unique(cids)
        self.cell_mask = np.ones(self.cellids.max()+1, dtype='i8')*-1
        self.cell_mask[self.cellids] = np.arange(len(self.cellids))
        print(len(self.cellids), 'cells from', self.cellids.min(), 'to', self.cellids.max())

    def _worker(self, rank, marray, sarray, numarray):
        module = rank % 16
        file_ind = rank // 16
        fname = sorted(glob.glob(EXPT_PREFIX + '/raw/r%.4d/*DSSC%.2d*.h5'%(self.run_num, module)))[file_ind]
        num_cells = len(self.cellids)

        mean = np.frombuffer(marray.get_obj(), dtype='f8').reshape(-1,16,num_cells,128,512)
        num_files = len(mean)
        mean = mean[file_ind,module]
        sigma = np.frombuffer(sarray.get_obj(), dtype='f8').reshape(-1,16,num_cells,128,512)[file_ind,module]
        num = np.frombuffer(numarray.get_obj(), dtype='i4').reshape(-1,16,num_cells)[file_ind,module]
        curr = (num, mean, sigma)

        stime = time.time()
        with h5py.File(fname, 'r') as f:
            dset = f['INSTRUMENT/'+DET_NAME+'/DET/%dCH0:xtdf/image/data'%module]
            tid = f['INSTRUMENT/'+DET_NAME+'/DET/%dCH0:xtdf/image/trainId'%module][:].ravel()
            cid = f['INSTRUMENT/'+DET_NAME+'/DET/%dCH0:xtdf/image/cellId'%module][:].ravel()
            
            num_chunks = int(np.ceil(dset.shape[0] / CHUNK_SIZE))
            for chunk in range(num_chunks):
                st, en = chunk*CHUNK_SIZE, (chunk+1)*CHUNK_SIZE
                cells = self.cell_mask[cid[st:en]]
                frames = dset[st:en,0,:,:].astype('f4')
                curr = self._update_stats(curr, frames, cells)
                if rank == 0:
                    sys.stderr.write('\r%d/%d chunks in %s (%f frames/s)' % (
                        chunk+1, num_chunks, 
                        os.path.basename(fname), 
                        num_files*(chunk+1)*CHUNK_SIZE/(time.time()-stime)))
                    sys.stderr.flush()
                #if chunk > 20:
                #    break
        if module == 0:
            sys.stderr.write('\n')
            sys.stderr.flush()
        sigma[:] = np.sqrt((sigma.transpose(1,2,0) / num).transpose(2,0,1))
        if module == 0:
            print(sigma.shape)
            sys.stderr.write('Updated data set, %f, %f\n'%(mean.mean(), sigma.mean()))

    def _update_stats(self, current, frames, cells):
        count, avg, m2 = current
        np.add.at(count, cells, 1)
        delta = frames - avg[cells]
        avg[cells] += (delta.transpose(1,2,0) / count[cells]).transpose(2,0,1)
        delta2 = frames - avg[cells]
        m2[cells] += delta * delta2
        return (count, avg, m2)

def main():
    parser = argparse.ArgumentParser(description='Process dark run')
    parser.add_argument('run', help='Run number', type=int)
    parser.add_argument('-m', '--mask', help='Recalculate masks from mean and sigma outliers', action='store_true')
    parser.add_argument('-o', '--out_fname', help='Output file name (input for masking)')
    args = parser.parse_args()

    proc = ProcDarks(args.run, mask_only=args.mask, out_fname=args.out_fname)
    proc.run()

if __name__ == '__main__':
    main()
