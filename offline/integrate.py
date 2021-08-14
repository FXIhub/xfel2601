import sys
import os.path as op
import time
import argparse
import multiprocessing as mp
import ctypes
import itertools

import numpy as np
import h5py
from mpi4py import MPI

PREFIX = '/gpfs/exfel/exp/SQS/202102/p002601/'
CHUNK_SIZE = 128
ADU_PER_PHOTON = 5.
SIGNAL_ESTIMATE = 1.e-3

class Integrator():
    def __init__(self, run, selector,
                 dark_run, testing=False,
                 num_frames=-1, num_cells=352):
        self.run = run
        self.dark_run = dark_run
        self.testing = testing
        self.num_frames = num_frames
        self.num_cells = num_cells

        if isinstance(selector, str):
            self.have_flag = True
            with h5py.File(selector, 'r') as f:
                self.flags = f['entry_1/do_integrate'][:].astype(np.bool)
        else:
            self.have_flag = False
            self.good_cells = selector
            self.num_cells = len(self.good_cells)

        self.f_vds = h5py.File(PREFIX + 'scratch/vds/r%.4d.cxi'%self.run, 'r')
        self.dset_vds = self.f_vds['entry_1/instrument_1/detector_1/data']
        if self.have_flag:
            assert self.flags.shape[0] == self.dset_vds.shape[0]

        if testing:
            self.out_fname = 'runsum_r%.4d'%self.run
        else:
            self.out_fname = PREFIX + 'scratch/gas/r%.4d'%self.run
        if self.have_flag:
            self.out_fname += '_sel'
        if self.num_frames > 0:
            self.out_fname += '_%.8d' % self.num_frames
        self.out_fname += '_integral.h5'

        if self.num_frames < 0:
            self.num_frames = self.dset_vds.shape[0]

    def finish(self, write=True):
        if write:
            with h5py.File(self.out_fname, 'w') as f:
                f['data/data'] = self.powder
                if not self.have_flag:
                    f['data/cells'] = np.where(self.good_cells)[0]
                f['data/num_frames'] = self.counts

        self.f_vds.close()

    def run_mpi(self):
        comm = MPI.COMM_WORLD
        rank = comm.rank
        nproc = comm.size
        if nproc % 16 != 0:
            raise ValueError('Need number of processes to be multiple of 16')
        if rank == 0:
            if self.have_flag:
                print('Processing %d/%d selected events' % (self.flags.sum(), self.flags.size))
            else:
                print('Processing %d/%d cells' % (self.good_cells.sum(), self.good_cells.size))
            print('Will write output to', self.out_fname)
            sys.stdout.flush()

        my_module = rank % 16
        psize = self.num_frames // (nproc // 16) + 1
        my_portion = np.arange(psize*(rank//16), min(self.num_frames, ((rank//16)+1)*psize))
        #print(rank, my_portion.min(), my_portion.max())

        with h5py.File(PREFIX + 'scratch/dark/r%.4d_dark.h5'%self.dark_run, 'r') as f:
            dark = f['data/mean'][my_module]
            dcells = f['data/cellId'][:]
            thresh = 0.5 - (f['data/sigma'][my_module]/ADU_PER_PHOTON)**2 * np.log(SIGNAL_ESTIMATE)

        num_chunks = len(my_portion) // CHUNK_SIZE + 1
        if self.testing:
            num_chunks = 10

        my_powder = np.zeros((16,128,512))
        my_counts = np.zeros(16, dtype='i4')

        stime = time.time()

        for chunk in range(num_chunks):
            st = chunk*CHUNK_SIZE
            en = min(len(my_portion), (chunk+1)*CHUNK_SIZE)
            chunk_ind = my_portion[st:en]
            if self.have_flag:
                chunk_ind = chunk_ind[self.flags[chunk_ind]]
                if len(chunk_ind) == 0:
                    continue
            cells = chunk_ind % self.num_cells
            if not self.have_flag and self.good_cells[cells].sum() == 0:
                continue

            chunk_cid = self.f_vds['entry_1/cellId'][chunk_ind, my_module]
            cellinds = np.array([np.where(dcells==cid)[0][0] for cid in chunk_cid])

            fr = self.dset_vds[chunk_ind, my_module, 0, :, :]
            fr = fr.astype('f4') - dark[cellinds]
            # Common mode correction
            fr = np.array([self._common_mode(mod) for mod in fr])
            # Fixed threshold
            #phot = np.round(fr/ADU_PER_PHOTON - 0.3).astype('i4')
            # Sigma-based threshold
            phot = np.ceil(fr/ADU_PER_PHOTON - thresh[cellinds]).astype('i4')
            if not self.have_flag:
                phot = phot[self.good_cells[cells]]
            phot = phot[~np.all(phot<0, axis=(1,2))]

            if phot.shape[0] == 0:
                continue

            phot[phot<0] = 0
            my_powder[my_module] += phot.sum(0)
            my_counts[my_module] += phot.shape[0]

            if rank == 4:
                sys.stderr.write('\r%d/%d: %d (%f Hz)' % (chunk+1, num_chunks, my_counts[my_module], (nproc//16)*(chunk+1)*CHUNK_SIZE/(time.time()-stime)))
                sys.stderr.flush()

        if rank == 4:
            sys.stderr.write('\nReducing\n')
            sys.stderr.flush()

        self.counts = np.zeros(16, dtype='i4')
        comm.Reduce(my_counts, self.counts, op=MPI.SUM, root=0)
        if rank == 0:
            sys.stderr.write('Reduced counts\n')
            sys.stderr.flush()

        self.powder = np.zeros((16,128,512), dtype='f8').flatten()
        for m in range(16):
            comm.Reduce(my_powder.flatten()[m*512*128:(m+1)*512*128], self.powder[m*512*128:(m+1)*512*128], op=MPI.SUM, root=0)
            if rank == 0:
                sys.stderr.write('Reduced powder %d\n' % m)
                sys.stderr.flush()

        if rank == 0:
            self.powder = self.powder.reshape((16,128,512))
            self.powder /= self.counts[:,np.newaxis,np.newaxis]
        self.finish(write=(rank==0))
        if rank == 0:
            sys.stderr.write('Wrote file\n')
            sys.stderr.flush()

    @staticmethod
    def _iterating_median(v, tol=3):
        if len(v) == 0:
            return 0
        vmin, vmax = v.min(), v.max()
        #vmin, vmax = -2*tol, 2*tol
        vmed = np.median(v[(vmin < v) & (v < vmax)])
        vmed0 = vmed
        i = 0
        while True:
            vmin, vmax = vmed-tol, vmed+tol
            vmed = np.median(v[(vmin < v) & (v < vmax)])
            if vmed == vmed0:
                break
            else:
                vmed0 = vmed
            i += 1
            if i > 20:
                break
        return vmed

    def _common_mode(self, img):
        """img should be subtracted by the dark.
        img.shape == (X, Y) 
        There is no mask
        The correction is applied IN-PLACE
        """
        ig = img.astype('f8').copy()
        L = 64
        for i, j in itertools.product(range(ig.shape[0] // L),
                                      range(ig.shape[1] // L)):
            img = ig[i*64:(i+1)*64, j*64:(j+1)*64]
            med = self._iterating_median(img.flatten())
            img -= med
        return ig

def main():
    parser = argparse.ArgumentParser(description='Calculate run integral')
    parser.add_argument('run', help='Run number', type=int)
    parser.add_argument('dark_run', help='Process raw data with this dark run', type=int, default=-1)
    parser.add_argument('-c', '--cells', help='Cell range to integrate (default: all)', default='all')
    parser.add_argument('-f', '--flag_file', help='Path to file containing flags of which events to process')
    parser.add_argument('-t', '--testing', help='Testing mode (only 10 chunks)', action='store_true')
    parser.add_argument('-n', '--num_trains', help='Only integrate the first N trains', type=int, default=-1)
    parser.add_argument('--num_cells', help='Set if number of detector cells is not 400', type=int, default=400)
    args = parser.parse_args()

    if args.flag_file is not None:
        good_cells = args.flag_file
    else:
        if args.cells == 'all':
            cells = [0, args.num_cells, 1]
        else:
            cells = [int(n) for n in args.cells.split(',')]
            if len(cells) < 2:
                raise ValueError('Need at least start and end values for cell range')
            if len(cells) < 3:
                cells = cells + [1]

        good_cells = np.zeros(args.num_cells, dtype='bool')
        good_cells[cells[0]:cells[1]:cells[2]] = True

    integ = Integrator(args.run, selector=good_cells,
                       dark_run=args.dark_run, num_frames=args.num_trains*args.num_cells,
                       testing=args.testing, num_cells=args.num_cells)
    integ.run_mpi()

if __name__ == '__main__':
    main()
