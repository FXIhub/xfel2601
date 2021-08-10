#!/usr/bin/env python

import h5py
import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib import colors
import numpy as np
import extra_geom
import spimage
from scipy import optimize

dirpath = '/gpfs/exfel/exp/SQS/202131/p900210/scratch/ayyerkar/events/'

vdspath = '/gpfs/exfel/exp/SQS/202131/p900210/scratch/ayyerkar/vds_2942/'

def get_geom():
    size_x = 519
    size_y = 497
    # First geometry refinement [01.08.2021]
    #quad_pos_pix = np.array([(-size_x-6, 0+8), (-size_x-15,-size_y-5), (0+4,-size_y-13), (0+14, 0-4)])*0.236
    # Updated geometry [02.08.2021]
    quad_pos_pix = np.array([(-size_x-3, 0+12), (-size_x-15,-size_y-5), (0+2,-size_y-15), (0+10, 0)])*0.236
    geom = extra_geom.DSSC_1MGeometry.from_h5_file_and_quad_positions('/gpfs/exfel/exp/SQS/202102/p002601/scratch/det/dssc_geom_AS_aug20.h5', quad_pos_pix)
    return geom

def get_dark():
    f = h5py.File('/gpfs/exfel/exp/SQS/202131/p900210/scratch/ayyerkar/data/dark_r0034.h5','r')
    dark = f['/data/mean']
    cells = f['/data/cellId']
    sigma = f['data/sigma'][:]
    return dark, cells, sigma

def get_radavg(intrad, data, mask):
    radcount = np.zeros(intrad.max()+1)
    radavg = np.zeros_like(radcount)
    mymask = mask & ~np.isnan(data)
    np.add.at(radcount, intrad[mymask], 1)
    np.add.at(radavg, intrad[mymask], data[mymask])
    return radavg / radcount

def sphere_q4(qvals, scale, dia):
    fluence_factor = 275821828
    s = qvals*np.pi*dia
    return scale*(dia**3*(np.sin(s) - s*np.cos(s)) / s**3)**2 * qvals**4 / fluence_factor

def sphere(qvals, scale, dia):
    fluence_factor = 275821828
    s = qvals*np.pi*dia
    return scale*(dia**3*(np.sin(s) - s*np.cos(s)) / s**3)**2 / fluence_factor


run = 47
f = h5py.File(dirpath+'r%04d_litpix.h5' % (run),'r')
litpixels = np.array(f['/entry_1/litpixels'])
litpixels = np.sum(litpixels,axis=0)
# Sorted in descending order
idxs = np.argsort(litpixels)[::-1]

f.close()

geom = get_geom()
dark, cells, sigma = get_dark()
mask = ~(sigma.mean(1) < 0.5) | (sigma.mean(1) > 1.5)

x, y, z = geom.get_pixel_positions().transpose(3,0,1,2) / 236e-6
intrad = np.sqrt(x*x + y*y).astype('i4')

q = 2*1.2/1.23984*np.sin(0.5*np.arctan(np.sqrt(x*x + y*y)*0.236/536))
qvals = 2*1.2/1.23984*np.sin(0.5*np.arctan(np.arange(intrad.max()+1)*0.236/536))

wavelength = 1.2398/1.2*1e-9

# Get sphere models for correlation
dvals = np.arange(30,150,0.5)
smodels = np.array([sphere_q4(qvals[40:140], 1e-7, d) for d in dvals])

f = h5py.File(vdspath+'r%04d.cxi' % (run),'r')

for idx in idxs[15:]:
   data = np.array(f['/entry_1/instrument_1/detector_1/data'][idx,:,0]).astype('float32')
   cell = f['/entry_1/cellId'][idx,0]
   print(cell)
   print(dark.shape)
   data -= dark[:,cell]
   print("r%04d: Image %d Hitscore = %d" % (run, idx, litpixels[idx]))
   fig, axs = plt.subplots(1,2,figsize=(20,8))
   ass,_ = geom.position_modules_fast(data)
   plt.title("Hitscore")

   axs[0].imshow(ass[300:900,300:800],norm=colors.LogNorm(vmin=2),aspect=177/204.38)

   radavg = get_radavg(intrad, data, mask)
   corr = np.corrcoef((radavg*qvals**4)[40:140], smodels)[0,1:]
   dia = dvals[corr.argmax()]
   print(dia)
   I_fit = lambda qval,I0: sphere_q4(qval,I0, dia)
   try:
       I0 = optimize.curve_fit(I_fit, qvals[40:140], (radavg*qvals**4)[40:140], p0=(1.,))[0]
   except:
       continue
   plt.suptitle('r%04d: Image %d Hitscore = %d' % (run, idx, litpixels[idx]))
   
   axs[1].semilogy(radavg)
   
   axs[1].semilogy(sphere(qvals, I0, dia))
   plt.title("Diameter = %d nm I0 = %d uJ/um^2" % (dia,I0))

   plt.subplots_adjust(wspace=0, hspace=0)
   plt.show()
