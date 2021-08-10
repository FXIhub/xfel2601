"""
For testing the EuXFEL backend, start the karabo server:

./karabo-bridge-server-sim 1234
    OR
./karabo-bridge-server-sim -d AGIPDModule -r 1234

from the karabo-bridge (https://github.com/European-XFEL/karabo-bridge-py).
and then start the Hummingbird backend:

./hummingbird.py -b examples/euxfel/mock/conf.py
"""
import plotting.image
import plotting.line
# import analysis.agipd
import analysis.event
import analysis.hitfinding
import ipc.mpi
from backend import add_record

import sys
sys.path.append("/gpfs/exfel/exp/SQS/202131/p900210/usr/Shared/xfel2601")
import tools.sizelib
import itertools

# Testing
# from backend.euxfel import ureg

import numpy as np
import time
import sys, os; sys.path.append(os.path.split(__file__)[0])
import h5py

# from online_agipd_calib import AGIPD_Calibrator

state = {}
# state['Facility'] = 'EuXFELtrains'
state['Facility'] = 'EuXFELDSSC'
# state['EuXFEL/EventIsTrain'] = True
state['EventIsTrain'] = True

# state['EuXFEL/DataSource'] = 'tcp://10.253.0.142:6666' # Calibrated

flag_online = True
flag_cm_correction = True

if flag_online:
    state['EuXFEL/DataSource'] = "tcp://10.253.0.190:47000"
else:
    state['EuXFEL/DataSource'] = 'tcp://localhost:9898' # From file (on exflonc35)

state['EuXFEL/DataFormat'] = 'Calib' # This is very specific to AGIPD, should use a new translator and remove this.
state['EuXFEL/MaxTrainAge'] = 4e10000000
#state['EuXFEL/MaxPulses'] = 176

state['EuXFEL/FirstCell'] = 0

# Use SelModule = None or remove key to indicate a full detector
# [For simulator, comment if running with full detector, otherwise uncomment]
# state['EuXFEL/SelModule'] = 0

PREFIX = '/gpfs/exfel/exp/SQS/202102/p002601/'
detector_distance = 0.55
wavelength = 1.03e-9
pixel_size = 204e-6
n_pulses = 2
adu_threshold = 4
hitscore_threshold = 1000
module_numbers = [0,7,8,15]

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

def _common_mode(img):
    """img should be subtracted by the dark.
    This version applies on all pixels (no mask)
    img.shape == (X, Y)"""
    ig = img.astype('f8').copy()
    L = 64
    for i, j in itertools.product(range(ig.shape[0] // L),
                                  range(ig.shape[1] // L)):
        img = ig[i*64:(i+1)*64, j*64:(j+1)*64]
        med = _iterating_median(img.ravel())
        img -= med
    return ig

def assem_00(mods, cellind):
    '''Assembly of inner modules from geom_00 coordinates
    mods is a list of 4 (512,400,ncells) arrays
    '''
    #assem = np.full((289,283), np.nan)
    assem = np.zeros((289,283))
    assem[161:161+128,140-128:140] = mods[0][:128,:,cellind].T[:,::-1]
    assem[ 13: 13+128,128-128:128] = mods[1][:128,:,cellind].T[:,::-1]
    assem[128-128:128,146:146+128] = mods[2][:128,:,cellind].T[::-1]
    assem[274-128:274,154:154+128] = mods[3][:128,:,cellind].T[::-1]
    return assem[::-1,::-1]

with h5py.File(PREFIX+'usr/Shared/aux/badpixel_mask_r0002.h5', 'r') as f:
    goodpix = f['entry_1/good_pixels'][:][[module_numbers]].transpose(0,2,1)
    goodpix_stacked = np.concatenate(tuple(goodpix), axis=1)

def onEvent(evt):
    #print(list(evt.keys()))
    #print(evt['photonPixelDetectors'].keys())
    #sys.stdout.flush()
    if not 'DSSC00' in evt['photonPixelDetectors']:
        print("No DSSC data, skipping event")
        return
    proc_rate = analysis.event.printProcessingRate(pulses_per_event=1)
    proc_rate = add_record(evt["analysis"], "analysis", "processing Rate", proc_rate)

    #print(evt['photonPixelDetectors']['DSSC00'].data.shape)
    mods = [evt['photonPixelDetectors']['DSSC%.2d'%m].data for m in module_numbers]

    stacked = np.concatenate(tuple(mods), axis=1)
    #stacked *= goodpix_stacked[:,:,np.newaxis]
    #stacked = np.concatenate(tuple([_common_mode(mod) for m in mods]), axis=1)
    #print(stacked.shape)
    modules = add_record(evt["analysis"], "analysis", "single", stacked[...,:n_pulses])

    analysis.hitfinding.countLitPixels(evt, modules, 
                                       aduThreshold=adu_threshold, 
                                       hitscoreThreshold=hitscore_threshold, 
                                       mask=goodpix_stacked[:,:,np.newaxis],
                                       stack=True)
    hitscore = evt["analysis"]["litpixel: hitscore"].data
    hittrain = np.bool8(evt["analysis"]["litpixel: isHit"].data)
    hitscore_pulse = add_record(evt["analysis"], "analysis", "hitscore", hitscore)
    for i in range(modules.data.shape[-1]):
        hitscore_pulse = add_record(evt["analysis"], "analysis", "hitscore", hitscore[i])
        plotting.line.plotHistory(hitscore_pulse, group="Hitfinding", hline=hitscore_threshold, history=10000)
    '''

    #print("modules shape",modules.data.shape)
    #print("hittrain shape",hittrain.shape)
    for hit_index in np.arange(len(hittrain))[hittrain]:
        single_hit = add_record(evt["analysis"], "analysis", "Hit", modules.data[...,hit_index])
        plotting.image.plotImage(single_hit, history=10)
    '''

    random_cell = 0 # Not really random, eh?
    random_image = add_record(evt["analysis"], "analysis", "Random Image", modules.data[...,random_cell])
    plotting.image.plotImage(random_image, history=10)

    squeezed_image = add_record(evt["analysis"], "analysis", "Squeezed Image", modules.data[...,random_cell])
    plotting.image.plotImage(squeezed_image, history=10, aspect_ratio=0.866)

    assem_image = add_record(evt["analysis"], "analysis", "Assem Image", assem_00(mods, random_cell))
    plotting.image.plotImage(assem_image, history=10, aspect_ratio=2/np.sqrt(3))

    if(proc_rate is not None):
        plotting.line.plotHistory(proc_rate, group="Hitfinding")
    return
