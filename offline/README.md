# Offline analysis for p002601
Here are some notes on analyzing data from p002601

## Environment
We are using a custom conda environment for our analysis. Run the following command to get the environment:
```
source source_this_at_euxfel
```
in the root directory of this repository.

## Primary pipeline
The pipeline will be based on processing runs using a series of python scripts over SLURM. 

The SLURM scripts in `offline/slurm/` are to be run in the following order on a given signal run:

 1. `vds_array.sh` - Generate virtual dataset (VDS) files for runs which virtually combine and synchronize the different DSSC modules
 2. `litpixels_array.sh` - Calculate number of pixels in each frame with at least 1 photon in the inner region of the detector
 3. `save_hits.sh` - Save the hits from the litpixels metric into an EMC-format file for additional analysis
 4. `crop_dragonfly.sh` - Generate 'lowq' and 'medq' EMC files to analyze only inner parts of detector
 
For dark runs:

 1. `proc_darks_array.sh` - Generate dark calibration constants (no VDS needed)

For sucrose runs:

After having run `litpixels_array.sh` on a sucrose run you can do the following steps for a size of fluence estimate from the hits:

 1. `radialavg_array.sh` - Calculates radial averages for all the hits.
 2. `sizing_fast.sh` - Fit sizes and estimate fluences and anisotropy (lack of sphericity) for all the hits.

The dark runs need to be processed before running any of the steps past Step 2 for signal runs

## Data explorer
A short wrapper class is provided to make it convenient to explore the data interactively here:
```
offline/explorer.py
```
This class can be used to look at data and come up with further analyses, which can then be incorporated into scripts for batch-processing.

## Dragonfly classification and analysis
To perform reconstructions using Dragonfly, first create a reconstruction directory
```
$ module load dragonfly
$ dragonfly_init -t <tag>
```
This creates a template directory which can then be used as usual. In order to submit jobs on SLURM, copy the following template
```
offline/slurm/dragonfly_template.sh
```
to the reconstruction directory. Remember to edit the number of nodes, time limit etc. before submitting.
