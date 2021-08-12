'''
For testing the EuXFEL backend, start the karabo server:

./karabo-bridge-server-sim 1234
    OR
./karabo-bridge-server-sim -d AGIPDModule -r 1234

from the karabo-bridge (https://github.com/European-XFEL/karabo-bridge-py).
and then start the Hummingbird backend:

./hummingbird.py -b examples/euxfel/mock/conf.py
'''
import plotting.image
import plotting.line
import plotting.correlation
# import analysis.agipd
import analysis.event
import analysis.hitfinding
import ipc.mpi
from backend import add_record

import sys
sys.path.append('/gpfs/exfel/exp/SQS/202131/p900210/usr/Shared/xfel2601')
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
    state['EuXFEL/DataSource'] = 'tcp://10.253.0.190:47000'
else:
    state['EuXFEL/DataSource'] = 'tcp://localhost:9898' # From file (on exflonc35)

state['EuXFEL/DataFormat'] = 'Calib' # This is very specific to AGIPD, should use a new translator and remove this.
state['EuXFEL/MaxTrainAge'] = 4e10000000
#state['EuXFEL/MaxPulses'] = 176

state['EuXFEL/FirstCell'] = 0

# Use SelModule = None or remove key to indicate a full detector
# [For simulator, comment if running with full detector, otherwise uncomment]
# state['EuXFEL/SelModule'] = 0

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
    '''img should be subtracted by the dark.
    This version applies on all pixels (no mask)
    img.shape == (X, Y)'''
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

def get_radavg(photons):
    radavg = np.zeros_like(radcount)
    np.add.at(radavg, intrad_stacked[goodpix_stacked], photons[goodpix_stacked])
    return radavg / radcount

def sphere_q4(qvals, fluence, dia):
    s = qvals * np.pi * dia
    return (fluence/fluence_factor) * (dia**3 * (np.sin(s) - s*np.cos(s)) / s**3)**2 * qvals**4

def fit_sphere(radavg):
    radavg_q4 = radavg * qvals**4
    corr = np.corrcoef(radavg_q4[fit_radmin:fit_radmax], sphere_models)[0,1:]
    dia_best = dia_vals[corr.argmax()]
    try:
        return optimize.curve_fit(sphere_q4,
                                  qvals[fit_radmin:fit_radmax],
                                  radavg_q4[fit_radmin:fit_radmax],
                                  p0=(1., dia_best))[0]
    except RuntimeError:
        return np.array([0., 0.])

PREFIX = '/gpfs/exfel/exp/SQS/202102/p002601/'
detector_distance = 0.536
wavelength = 1.03e-9
pixel_size = 236e-6
fluence_factor = 275821828 # For this wavelength and for sucrose
#n_pulses = 4
adu_threshold = 4
adu_per_photon = 5.
module_numbers = [0,7,8,15]
fit_radmin = 30
fit_radmax = 100
dia_vals = np.arange(5,80,0.5)
hitscore_threshold = 13000
strong_hit_threshold = 25000
pulse_slice = slice(0,100,1)

do_hitfinding = True
do_common_mode = False
do_show_random = True
do_sizing = True
send_motors = True
do_aduhist = True

with h5py.File(PREFIX+'usr/Shared/aux/badpixel_mask_r0002.h5', 'r') as f:
    goodpix = f['entry_1/good_pixels'][:][[module_numbers]].transpose(0,2,1)
    goodpix_stacked = np.concatenate(tuple(goodpix), axis=1)

with h5py.File(PREFIX+'usr/Shared/aux/r0062_dark.h5', 'r') as f:
    dark = f['data/mean'][:]

sys.path.append(PREFIX+'usr/Shared/aux/')
import geom_00
geom = geom_00.get_geom()
x, y, _ = geom.get_pixel_positions()[module_numbers].transpose(3,0,2,1) / 236e-6
intrad = np.sqrt(x*x + y*y).astype('i4')
intrad_stacked = np.concatenate(tuple(intrad), axis=1)
radcount = np.zeros(intrad_stacked.max()+1)
np.add.at(radcount, intrad_stacked[goodpix_stacked], 1)
radcount[radcount==0] = 1 # Avoiding NaNs for now
qvals = 2/(wavelength*1e9) * \
        np.sin(0.5*np.arctan(np.arange(intrad_stacked.max()+1)*pixel_size/detector_distance))
sphere_models = np.array([sphere_q4(qvals[fit_radmin:fit_radmax], 1, dia) for dia in dia_vals])

def onEvent(evt):
    #print(list(evt.keys()))
    #print(evt['photonPixelDetectors'].keys())
    #print(evt.native_keys())
    if not 'DSSC00' in evt['photonPixelDetectors']:
        print('No DSSC data, skipping event')
        return
    proc_rate = analysis.event.printProcessingRate(pulses_per_event=1)
    proc_rate = add_record(evt['analysis'], 'analysis', 'processing Rate', proc_rate)

    #print(evt['photonPixelDetectors']['DSSC00'].data.shape)
    #mods = [evt['photonPixelDetectors']['DSSC%.2d'%m].data[...,pulse_slice] - dark[m,pulse_slice,:,:].transpose(2,1,0) for m in module_numbers]
    mods = [evt['photonPixelDetectors']['DSSC%.2d'%m].data[...,pulse_slice] for m in module_numbers]

    if do_common_mode:
        stacked = np.concatenate(tuple([_common_mode(mod) for m in mods]), axis=1)
    else:
        stacked = np.concatenate(tuple(mods), axis=1)
    #stacked *= goodpix_stacked[:,:,np.newaxis]
    #print(stacked.shape)
    modules = add_record(evt['analysis'], 'analysis', 'single', stacked)

    if do_hitfinding:
        analysis.hitfinding.countLitPixels(evt, modules,
                                           aduThreshold=adu_threshold,
                                           hitscoreThreshold=hitscore_threshold,
                                           mask=goodpix_stacked[:,:,np.newaxis],
                                           stack=True)
        hitscore = evt['analysis']['litpixel: hitscore'].data
        hittrain = np.bool8(evt['analysis']['litpixel: isHit'].data)
        analysis.hitfinding.hitrate(evt, hittrain)
        if ipc.mpi.is_main_worker():
            plotting.line.plotHistory(evt['analysis']['hitrate'], label='Hit rate [%]', group='Hitfinding')
        hitscore_pulse = add_record(evt['analysis'], 'analysis', 'hitscore', hitscore)
        for i in range(modules.data.shape[-1]):
            hitscore_pulse = add_record(evt['analysis'], 'analysis', 'hitscore', hitscore[i])
            plotting.line.plotHistory(hitscore_pulse, group='Hitfinding', hline=hitscore_threshold, history=10000)

        if hittrain.sum() > 0: # If at least one cell is a hit, send brightest hit
            hit_index = hitscore.argmax()
            brightest_hit = add_record(evt['analysis'], 'analysis', 'Brightest Hit', assem_00(mods,hit_index))
            plotting.image.plotImage(brightest_hit, group='Hitfinding', history=10)

        if hittrain.sum() > 0: # If at least one cell is a hit, send weakest hit
            hit_index = np.where(hittrain)[0][hitscore[hittrain].argmin()]
            brightest_hit = add_record(evt['analysis'], 'analysis', 'Weakest Hit', assem_00(mods,hit_index))
            plotting.image.plotImage(brightest_hit, group='Hitfinding', history=10)

    if do_show_random:
        random_cell = 0 # Not really random, eh?
        random_image = add_record(evt['analysis'], 'analysis', 'Random Image', modules.data[...,random_cell])
        plotting.image.plotImage(random_image, history=10, group='Random')

        squeezed_image = add_record(evt['analysis'], 'analysis', 'Squeezed Image', modules.data[...,random_cell])
        plotting.image.plotImage(squeezed_image, history=10, aspect_ratio=0.866, group='Random')

        assem_image = add_record(evt['analysis'], 'analysis', 'Assem Image', assem_00(mods, random_cell))
        plotting.image.plotImage(assem_image, history=10, aspect_ratio=2/np.sqrt(3), group='Random')

    if do_hitfinding and do_sizing:
        strong_hittrain = evt['analysis']['litpixel: hitscore'].data > strong_hit_threshold
        for hit_index in np.where(strong_hittrain)[0]:
            photons = np.round(modules.data[..., hit_index] / adu_per_photon - 0.3)
            photons[photons < 0] = 0
            radavg = get_radavg(photons)
            fluence, dia = fit_sphere(radavg)
            if dia > 0.:
                dia_record = add_record(evt['analysis'], 'analysis', 'sizing: diameter', dia)
                fluence_record = add_record(evt['analysis'], 'analysis', 'sizing: fluence', fluence)
                plotting.line.plotHistory(dia_record, group='Sizing')
                plotting.line.plotHistory(fluence_record, group='Sizing')

    if send_motors:
        inj_x = add_record(evt['analysis'], 'analysis', 'injector_x', evt._evt['SQS_AQS_MOLB/MOTOR/NOZZLE_X']['actualPosition.value'])
        inj_y = add_record(evt['analysis'], 'analysis', 'injector_y', evt._evt['SQS_AQS_MOLB/MOTOR/NOZZLE_Y']['actualPosition.value'])
        inj_z = add_record(evt['analysis'], 'analysis', 'injector_z', evt._evt['SQS_AQS_MOLB/MOTOR/NOZZLE_Z']['actualPosition.value'])
        plotting.line.plotHistory(inj_x, group='Motors', history=1000)
        plotting.line.plotHistory(inj_y, group='Motors', history=1000)
        plotting.line.plotHistory(inj_z, group='Motors', history=1000)
        if do_hitfinding and ipc.mpi.is_main_worker():
            plotting.correlation.plotScatter(inj_x, evt['analysis']['hitrate'], xlabel='Hitrate', ylabel='Inj-X', group='Hitfinding')

    if do_aduhist:
        #hbins = add_record(evt['analysis'], 'analysis', 'adu_hist_bins', np.arange(-5,20,0.02)[:-1]+0.01)
        #hist = add_record(evt['analysis'], 'analysis', 'adu_hist', np.histogram(stacked[...,0], bins=np.arange(-5,20,0.02))[0])
        #plotting.line.plotTrace(paramY=hist, paramX=hbins, label='ADU histogram', history=100)
        for i in range(modules.data.shape[-1]):
            pixval = add_record(evt['analysis'], 'analysis', 'pixel_val', stacked[10,10,i])
            plotting.line.plotHistory(pixval, group='ADU Histogram', history=100000)

    if proc_rate is not None:
        plotting.line.plotHistory(proc_rate, group='Hitfinding')
