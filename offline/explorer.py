'''Module to ease exploration of data from VDS files
Only for interactive use (Don't build scripts using this!!)

Import in IPython or Jupyter
Use appropriate matplotlib magic command for plotting
'''

import sys
import importlib
import itertools
import warnings

import numpy as np
import pylab as P
import h5py

PREFIX = '/gpfs/exfel/exp/SQS/202102/p002601/scratch/'

sys.path.append(PREFIX+'det')

class Explorer():
    def __init__(self, run, geom_file='geom_00'):
        self._fvds = None
        self.open_run(run)

        self.geom = importlib.import_module(geom_file).get_geom()
        x, y, _ = self.geom.get_pixel_positions().transpose(3,0,1,2) / 236e-6
        self.intrad = np.sqrt(x*x + y*y).astype('i4')
        self.radcount = np.zeros(self.intrad.max()+1)

    def open_run(self, run):
        self.run_num = run
        if self._fvds is not None:
            self._fvds.close()
        self._fvds = h5py.File(PREFIX+'vds/r%.4d.cxi'%run, 'r')
        self._dset = self._fvds['entry_1/instrument_1/detector_1/data']
        print('VDS data set shape:', self._dset.shape)

    def parse_dark(self, dark_run):
        '''Get dark offsets and bad pixel mask'''
        with h5py.File(PREFIX+'dark/r%.4d_dark.h5'%dark_run, 'r') as f:
            self.dark = f['data/mean'][:]
            self.dcells = f['data/cellId'][:]
            sigma = f['data/sigma'][:]
        self.mask = (sigma.mean(1) < 0.5) | (sigma.mean(1) > 1.5)

    @staticmethod
    def assemble_dense(frame, out=None):
        '''Frame assembly with no panel gaps
        Use imshow(out, origin='lower') to plot properly
        '''
        assert frame.shape == (16,128,512)
        if out is None:
            out = np.zeros((1024,1024), dtype=frame.dtype)
        out[512:,512:] = frame[:4].reshape(512,512)
        out[:512,512:] = frame[4:8].reshape(512,512)
        out[:512,:512] = frame[8:12][::-1,::-1,::-1].reshape(512,512)
        out[512:,:512] = frame[12:][::-1,::-1,::-1].reshape(512,512)
        return out

    def get_radavg(self, data):
        self.radcount[:] = 0.
        radavg = np.zeros_like(self.radcount)
        mymask = ~self.mask & ~np.isnan(data)
        np.add.at(self.radcount, self.intrad[mymask], 1)
        np.add.at(radavg, self.intrad[mymask], data[mymask])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return radavg / self.radcount

    @staticmethod
    def _iterating_median(v, tol=3):
        if len(v) == 0:
            return 0
        #vmin, vmax = v.min(), v.max()
        vmin, vmax = -2*tol, 2*tol
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

    def _common_mode(self, img, mask):
        """img should be substracted by the dark.
        img.shape == (X, Y), mask.shape == (X, Y)"""
        ig = img.astype('f8').copy()
        L = 64
        for i, j in itertools.product(range(ig.shape[0] // L),
                                      range(ig.shape[1] // L)):
            img = ig[i*64:(i+1)*64, j*64:(j+1)*64]
            m = mask[i*64:(i+1)*64, j*64:(j+1)*64]
            med = self._iterating_median(img[m].flatten())
            img -= med
        return ig

    def get_corr(self, i, cmod=False):
        if self.dark is None:
            raise AttributeError('Parse darks first to get corrected frame')
        cellid = self._fvds['entry_1/cellId'][i, 0]
        out = self._dset[i,:,0] - self.dark[:, np.where(cellid==self.dcells)[0][0]]
        if not cmod:
            return out
        return np.array([self._common_mode(out[i], ~self.mask[i]) for i in range(16)])

    def plot_frame(self, i, vmin=-3, vmax=10, cmod=False):
        frame = self.get_corr(i, cmod=cmod)
        assem, cen = self.geom.position_modules_fast(frame)
        P.imshow(assem[:,::-1], origin='lower', aspect=assem.shape[1]/assem.shape[0]*np.sqrt(3)/2, vmin=vmin, vmax=vmax)
        P.gca().set_facecolor('dimgray')
