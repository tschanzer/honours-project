# Data-driven parametrisation for Rayleigh-Bénard convection

This repository contains the code and Python notebooks associated with
an Honours thesis submitted by Thomas Schanzer at the University of New
South Wales in November 2023.

## Installation
To install the modules and their dependencies, first clone this repository:
```bash
$ git clone git@github.com:tschanzer/honours-project.git
$ cd honours-project
```
Then create the conda environment:
```bash
$ conda env create -f environment.yml
$ conda activate honours-project
```
You will then need to install my forked version of Dedalus v3, which
fixes a bug impacting the coarse-graining code, instead of the official
version:
```bash
$ conda install -c conda-forge dedalus c-compiler cython "h5py=*=mpi*"
$ conda uninstall --force dedalus
$ CC=mpicc pip3 install --no-cache --no-build-isolation http://github.com/tschanzer/dedalus/zipball/old-state/
```
Finally, install the custom modules using `pip`:
```bash
$ pip install -e .
```

## Obtaining the raw data
The ~109 GB of raw data needed to run the notebooks in
`honours-project/final_notebooks` and reproduce the figures in the
thesis are split across three Zenodo records with the following DOIs:

1. [10.5281/zenodo.10090744](https://doi.org/10.5281/zenodo.10090744):
    `resolution_tests_part1.zip`
2. [10.5281/zenodo.10090882](https://doi.org/10.5281/zenodo.10090882):
    `resolution_tests_part2.zip`
3. [10.5281/zenodo.10090902](https://doi.org/10.5281/zenodo.10090902):
    `training.zip`, `evaluation.zip`

In order to run the notebooks, you will need to extract these archives
into a new directory `honours-project/final_data`, merge the contents
of `resolution_tests_part1.zip` and `resolution_tests_part2.zip` into a
single directory named `resolution_tests`, and change all instances of
```python
base_path = '/srv/ccrc/AtmSS/z5310829/honours_project/'
```
in the notebooks to reflect the location of the repository on your
machine.

## Further information
For further information on the code and how to reproduce the data and
figures, refer to Appendix A of the thesis, which is publicly available at
https://github.com/tschanzer/honours-writing.
