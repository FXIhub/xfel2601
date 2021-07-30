import numpy as np
from .cpplib import radialAverage
from scipy.special import jn

def ballRadialIntensity(fluence, size, pixel):
    p = pixel * size
    return fluence * p ** (-3) * jn(1.5, 2*np.pi*p)**2


class _BallTemplate:
    def __init__(self):
        self._last_input = (None, None)
        self._result = None

    def __call__(self, rs, r):
        if not (np.all(rs == self._last_input[0]) and
                np.all(r == self._last_input[1])):
            self._last_input = (rs, r)
            ball_fft = np.zeros(( len(rs), len(r)))
            for i, rad in enumerate(rs):
                ball_fft[i] = ballRadialIntensity(1, rad, r)
            self._result = ball_fft
        return self._result
ballTemplate = _BallTemplate()


def sizeing(hits, mask, wavelength, detector_distance, pixel_size, center=(511.5, 511.5), rmin=10e-9, rmax=2000e-9, rstep=200):
    "dimension of hits and mask: (Y, X, cell Id)"
    s, c, r = radialAverage(*center, mask, hits, 1)
    r += 1e-4
    s /= c
    rs = np.linspace(rmin, rmax, rstep)
    r = 2*np.sin(0.5*np.arctan(r*pixel_size/detector_distance))/wavelength
    ball_inten = ballTemplate(rs, r)
    # ff = (s @ ball_inten.T) / np.linalg.norm(ball_inten, axis=1)                                                                           
    #print(s.shape, ball_inten.shape)                                                                                                        
    ff = (s @ ball_inten.T) / np.linalg.norm(ball_inten, axis=1)
    # ff = np.zeros((s.shape[0], ball_inten.shape[0]))                                                                                       
    # ball_inten = 1 / ball_inten                                                                                                            
    # for i in range(s.shape[0]):                                                                                                            
        # foo = s[i] * ball_inten                                                                                                            
        # ff[i] = np.std(foo, axis=1) / np.mean(foo, axis=1)                                                                                 
        # ff[i] = -np.std(foo, axis=1)                                                                                                       

    # ff = (s @ ball_inten.T) / np.linalg.norm(ball_inten, axis=1) / np.linalg.norm(s, axis=1)[:, np.newaxis]                                

    best_index = np.argmax(ff, axis=1)
    size_res = rs[best_index]
    intensity_res = (s*c / ball_inten[best_index]).mean(axis=1)
    return size_res, intensity_res, ff
