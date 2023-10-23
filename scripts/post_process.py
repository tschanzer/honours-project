"""Script for post-processing HDF5 output files."""

import argparse
import glob
import logging
import os

import dedalus.public as d3
import xarray as xr

logger = logging.getLogger(__name__)


def post_process(file):
    """
    Puts velocity and temperature data into a netCDF file.

    The output file will have the same name and location as the input.

    Args:
        file: Path to HDF5 data file.

    Returns:
        The path to the output file.
    """

    data = d3.load_tasks_to_xarray(file, tasks=['u', 'theta'])
    data = xr.Dataset(data)
    data['ux'] = data.u.sel({'': 0})
    data['uz'] = data.u.sel({'': 1})
    data = data.drop('u')
    data = data.rename({'ux': 'u', 'uz': 'w'})

    outdir = os.path.dirname(file)
    filename = os.path.basename(file).removesuffix('.h5') + '.nc'
    outfile = os.path.join(outdir, filename)
    data.to_netcdf(outfile)
    return outfile


if __name__ == '__main__':
    argParser = argparse.ArgumentParser(
        description=('Puts velocity and temperature data into netCDF files')
    )
    argParser.add_argument('dir', type=str, help='Path to data directory')
    args = argParser.parse_args()

    files = glob.glob(os.path.join(args.dir, '*.h5'))
    for i, f in enumerate(files):
        logger.info(f'[{i+1:d}/{len(files):d}]\tProcessing {f:s}')
        out = post_process(f)
        logger.info(f'[{i+1:d}/{len(files):d}]\tSaved {out:s}')
