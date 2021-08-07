#!/usr/bin/env python

import numpy as np
import spimage

# From https://journals.iucr.org/m/issues/2017/03/00/it5009/index.html
# Eq. 1 plus errata https://journals.iucr.org/m/issues/2019/03/00/it9021/index.html

# I_i = I_0 QE [(pi**2 d**3 |delta_n| delta_x)/(D lambda**2)]**2 |(sin(s_i) - s_i cos(s_i)) / s_i**3|**2

# Kartik's scale =  I_0 QE [(pi**2 |delta_n| delta_x)/(D lambda**2)]**2

# To convert it to I_0 we have:
# I_0 =  Kartik's scale * 1/(QE [(pi**2 |delta_n| delta_x)/(D lambda**2)]**2)

eV = 1200
material = 'sucrose'
mat = spimage.Material(material_type=material)

# delta n (refractive index)
d_n = (np.abs(1-mat.get_n(eV)))
# delta x (pixel size)
a = 236e-6/2
pixel_area = 3*np.sqrt(3)/2*a**2
# d_x is more accurately the square root of the pixel area (pixel_size assumes square pixels)
d_x = np.sqrt(pixel_area)

# detector distance in nm! (because Kartik used dimaters in nm)
D = 536e6
# Assuming QE or 1
QE = 1

# wavelength in nm (because Kartik used dimaters in nm)
W = 1240/eV

# We'll call K_SI = 1/(QE [(pi**2 |delta_n| delta_x)/(D lambda**2)]**2)
# which converts from Kartik's scale to photons/m**2
K_SI = 1./(QE*(np.pi**2 * d_n * d_x / (D*W**2))**2)
I0 = 1e-7 * K_SI

photon_enery_J = 1.602e-19*eV

# Now convert to uJ/um^2
K = K_SI*(photon_enery_J*1e6)*1e-12

Kartiks_scale_max = 1e-7
print("For %s and a maximum Kartik scale of %g we have a max I0:" % (material,Kartiks_scale_max))
print("I0 = %g photons/m^2" % (Kartiks_scale_max*K_SI))
print("I0 = %g uJ/um^2" % (Kartiks_scale_max*K))
print("Conversion constant to uJ/um^2 - %g" % K)
