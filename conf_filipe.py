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
    state['EuXFEL/DataSource'] = "tcp://10.253.0.143:41000"
else:
    state['EuXFEL/DataSource'] = 'tcp://localhost:9898' # From file (on exflonc35)

state['EuXFEL/DataFormat'] = 'Calib' # This is very specific to AGIPD, should use a new translator and remove this.
state['EuXFEL/MaxTrainAge'] = 4e10000000
#state['EuXFEL/MaxPulses'] = 176

state['EuXFEL/FirstCell'] = 0

# Use SelModule = None or remove key to indicate a full detector
# [For simulator, comment if running with full detector, otherwise uncomment]
# state['EuXFEL/SelModule'] = 0

detector_distance = 0.55
wavelength = 1.03e-9
pixel_size = 204e-6
n_pulses = 50
adu_threshold = 10
hitscore_threshold = 4000


f = h5py.File('/gpfs/exfel/exp/SQS/202131/p900210/usr/Shared/filipe/cal.1627753573.576151.h5','r')
dark_cal = np.array(f['/DSSC_MiniSDDV1_F2_004/Offset/0/data'])
print("dark cal",dark_cal.shape)


def onEvent(evt):
    print(list(evt.keys()))
    print(evt['photonPixelDetectors'])
    if not 'DSSC0' in evt['photonPixelDetectors']:
    # if True:
        print("No DSSC data, skipping event")
        return
    proc_rate = analysis.event.printProcessingRate(pulses_per_event=1)
    print(evt['photonPixelDetectors']['DSSC0'].data.shape)
    det = evt['photonPixelDetectors']['DSSC0'].data
    module = add_record(evt["analysis"], "analysis", "single", det[...,:n_pulses]-dark_cal[:,:,:n_pulses])
    proc_rate = add_record(evt["analysis"], "analysis", "processing Rate", proc_rate)

    analysis.hitfinding.countLitPixels(evt, module, aduThreshold=adu_threshold, hitscoreThreshold=hitscore_threshold, stack=True)
    hitscore = evt["analysis"]["litpixel: hitscore"].data
    hittrain = np.bool8(evt["analysis"]["litpixel: isHit"].data)
    hitscore_pulse = add_record(evt["analysis"], "analysis", "hitscore", hitscore)
    for i in range(module.data.shape[2]):
        hitscore_pulse = add_record(evt["analysis"], "analysis", "hitscore", hitscore[i])
        plotting.line.plotHistory(hitscore_pulse, group="Hitfinding", hline=hitscore_threshold, history=10000)

    print("module shape",module.data.shape)
    print("hittrain shape",hittrain.shape)
    for hit_index in np.arange(len(hittrain))[hittrain]:
        single_hit = add_record(evt["analysis"], "analysis", "Hit", module.data[...,hit_index])
        plotting.image.plotImage(single_hit, history=10)

    random_image = add_record(evt["analysis"], "analysis", "Random Image", module.data[...,10])
    plotting.image.plotImage(random_image, history=10)

    
    #plotting.image.plotImage(module, send_rate=10)
    if(proc_rate is not None):
        plotting.line.plotHistory(proc_rate, group="Hitfinding")
    return
