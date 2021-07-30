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

# from online_agipd_calib import AGIPD_Calibrator

state = {}
# state['Facility'] = 'EuXFELtrains'
state['Facility'] = 'EuXFELDSSC'
# state['EuXFEL/EventIsTrain'] = True
state['EventIsTrain'] = True

# state['EuXFEL/DataSource'] = 'tcp://10.253.0.142:6666' # Calibrated

flag_online = False
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

adu_threshold = 3
hitscore_threshold = 10

# Ugly assemble function.
def assemble(data, indices, shape=(1024, 1024), center=None):
    base_position = {0: (0, 0, False),
                     1: (0, 128, False),
                     2: (0, 2*128, False),
                     3: (0, 3*128, False),
                     4: (0, -4*128, False),
                     5: (0, -3*128, False),
                     6: (0, -2*128, False),
                     7: (0, -1*128, False),
                     8: (-512, -1*128, True),
                     9: (-512, -2*128, True),
                     10: (-512, -3*128, True),
                     11: (-512, -4*128, True),
                     12: (-512, 3*128, True),
                     13: (-512, 2*128, True),
                     14: (-512, 1*128, True),
                     15: (-512, 0*128, True)}
    if len(data[0].shape) == 3:
        shape += (data[0].shape[2], )
    array = np.zeros(shape)
    mask = np.zeros(shape, dtype="bool8")

    if center == None:
        center = [array.shape[0]//2, array.shape[1]//2]
    for panel, index in zip(data, indices):
        xllimit = max(center[0]+base_position[index][0], 0)
        xloffset = max(0, -(center[0]+base_position[index][0]))
        xulimit = min(center[0]+base_position[index][0]+panel.shape[0], array.shape[0])
        xuoffset = max(0, (center[0]+base_position[index][0]+panel.shape[0]) - array.shape[0])
        yllimit = max(center[1]+base_position[index][1], 0)
        yloffset = max(0, -(center[1]+base_position[index][1]))
        yulimit = min(center[1]+base_position[index][1]+panel.shape[1], array.shape[1])
        yuoffset = max(0, (center[1]+base_position[index][1]+panel.shape[1]) - array.shape[1])
        step = -1 if base_position[index][2] else 1
        
        if xllimit < xulimit and yllimit < yulimit:
            array[xllimit:xulimit, yllimit:yulimit] = panel[::step, ::step][xloffset:panel.shape[0]-xuoffset, yloffset:panel.shape[1]-yuoffset]
            mask[xllimit:xulimit, yllimit:yulimit] = True
    return array, mask


def onEvent(evt):
    # print(list(evt.keys()))
    # print(list(evt['photonPixelDetectors'].keys()))
    if not 'DSSC0' in evt['photonPixelDetectors']:
    # if True:
        print("No DSSC data, skipping event")
        return
    analysis.event.printProcessingRate(pulses_per_event=1)

    print(evt['photonPixelDetectors'].keys())
    
    # print(evt['photonPixelDetectors']['DSSC0'].data.shape)
    # print(evt['photonPixelDetectors']['DSSC7'].data.shape)
    # print(evt['photonPixelDetectors']['DSSC8'].data.shape)
    # print(evt['photonPixelDetectors']['DSSC15'].data.shape)

    
    if flag_online:
        data = [np.float64(evt['photonPixelDetectors']['DSSC0'].data),
                np.float64(evt['photonPixelDetectors']['DSSC7'].data)]
        module_index = [0, 7]
    else:
        # assembled_data, mask = assemble([evt['photonPixelDetectors']['DSSC0'].data.transpose(),
        #                                  evt['photonPixelDetectors']['DSSC7'].data.transpose()],
        #                                 [0, 7])
        data = [np.float64(evt['photonPixelDetectors'][f'DSSC{i}'].data.transpose()) for i in range(16)]
        module_index = list(range(16))

    if flag_cm_correction:
        for this_data in data:
            this_data[:, :64, :] -= np.median(this_data[:, :64, :], axis=1)[:, np.newaxis, :]
            this_data[:, 64:, :] -= np.median(this_data[:, 64:, :], axis=1)[:, np.newaxis, :]
            # this_data[:, :64, :] -= 100.
            # this_data[:, 64:, :] -= 50.
    
    assembled_data, mask = assemble(data, module_index)
    assembled = add_record(evt["analysis"], "analysis", "Assembled", assembled_data)


    analysis.hitfinding.countLitPixels(evt, assembled, aduThreshold=adu_threshold, hitscoreThreshold=hitscore_threshold, stack=True)
    hitscore = evt["analysis"]["litpixel: hitscore"].data
    hittrain = np.bool8(evt["analysis"]["litpixel: isHit"].data)
    
    for i in range(assembled.data.shape[2]):
        hitscore_pulse = add_record(evt["analysis"], "analysis", "hitscore", hitscore[i])
        plotting.line.plotHistory(hitscore_pulse, group="Hitfinding", hline=hitscore_threshold, history=10000)

    for hit_index in np.arange(len(hittrain))[hittrain]:
        single_hit = add_record(evt["analysis"], "analysis", "Hit", assembled.data[..., hit_index])
        plotting.image.plotImage(single_hit, mask=mask[..., hit_index], history=10)

    all_hits = assembled.data[..., hittrain]
    mask_hits = mask[..., hittrain]
    # for i in range(assembled.data.shape[-1]):
    #     assembled_single = add_record(evt["analysis"], "analysis", "Assembled Single", assembled.data[..., i])
    #     plotting.image.plotImage(assembled_single, mask=mask[..., i], history=10)

    size, intensity, ff = tools.sizelib.sizeing(all_hits, mask_hits, wavelength, detector_distance, pixel_size)
    # print(size)
    # print(intensity)
    # print(ff)
    for i in range(size.shape[-1]):
        size_record = add_record(evt["analysis"], "analysis", "Sizeing: size", size[i])
        intensity_record = add_record(evt["analysis"], "analysis", "Sizeing: intensity", intensity[i])
        plotting.line.plotHistory(size_record, history=10000, group="Sizeing")
        plotting.line.plotHistory(intensity_record, history=10000, group="Sizeing")
    
    # first_module = add_record(evt["analysis"], "analysis", "Single image", evt['photonPixelDetectors']['DSSC0'].data[..., 0])
    # plotting.image.plotImage(first_module, history=10, name="All images")
    
    # dssc = evt['photonPixelDetectors']['DSSC']
    # dssc_data = evt['photonPixelDetectors']['DSSC'].data
    
    # analysis.hitfinding.countLitPixels(evt, dssc, aduThreshold=adu_threshold, hitscoreThreshold=hitscore_threshold, stack=True)
    # # print(evt["analysis"]["litpixel: hitscore"].data)
    # hitscore = evt["analysis"]["litpixel: hitscore"].data
    # hittrain = np.bool8(evt["analysis"]["litpixel: isHit"].data)

    # first_hitscore = add_record(evt["analysis"], "analysis", "hitscore - first", hitscore[0])
    # plotting.line.plotHistory(first_hitscore, hline=hitscore_threshold, history=10000)

    # # for i in range(dssc_data.shape[0]):
    
    # print(dssc_data.shape)
    # for i in range(dssc_data.shape[0]):
    #     single_dssc = add_record(evt["analysis"], "analysis", "Single image", dssc_data[i, ...])
    #     plotting.image.plotImage(single_dssc, history=10, name="All images")

