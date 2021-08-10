import h5py
import numpy as np

PREFIX = '/gpfs/exfel/exp/SQS/202102/p002601/'

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Bad pixel mask generator')
    parser.add_argument('dark_run', type=int, help='Dark run number')
    parser.add_argument('-o', '--out_folder', 
                        help='Path of output folder (default=%s/scratch/det/)'%PREFIX,
                        default=PREFIX+'scratch/det/')
    args = parser.parse_args()

    with h5py.File(PREFIX + '/scratch/dark/r%.4d_dark.h5'% args.dark_run, 'r') as f:
        sigma = np.array(f['data/sigma'])
        mask = ~(sigma.mean(1) < 0.5) | (sigma.mean(1) > 1.5)

    with h5py.File(args.out_folder+"/"+"badpixel_mask_r%04d.h5" % args.dark_run, 'a') as outf:
        dset_name = 'entry_1/good_pixels'
        if dset_name in outf: del outf[dset_name]
        outf[dset_name] = mask
        dset_name = 'entry_1/bad_pixels'
        if dset_name in outf: del outf[dset_name]
        outf[dset_name] = ~mask
        
if __name__ == '__main__':
    main()
