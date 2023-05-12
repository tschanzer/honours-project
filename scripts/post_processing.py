"""Script for post-processing HDF5 output files."""

import dedalus.public as d3
import xarray as xr
import os
import argparse
import logging
import glob
import h5py
logger = logging.getLogger(__name__)

def post_process(file, coefficients=False):
    """
    Puts velocity and temperature data into a netCDF file.

    The output file will have the same name and location as the input.

    Args:
        file: Path to HDF5 data file.
        coefficients: Set to True to include coefficient data.

    Returns:
        The path to the output file.
    """

    data = d3.load_tasks_to_xarray(file, tasks=['u', 'theta'])
    data = xr.Dataset(data)
    data['ux'] = data.u.sel({'': 0})
    data['uz'] = data.u.sel({'': 1})
    data = data.drop('u')
    data = data.rename({'ux': 'u', 'uz': 'w'})

    if coefficients:
        # For some reason d3.load_tasks_to_xarray doesn't work for
        # the coefficient data, so we have to add it manually
        with h5py.File(file, mode='r') as f:
            coef_theta = f['tasks']['coef_theta'][:]
            coef_u = f['tasks']['coef_u'][:,0,:,:]
            coef_w = f['tasks']['coef_u'][:,1,:,:]

        coef_data = xr.Dataset(
            {
                'coef_theta': (['t', 'kx', 'kz'], coef_theta),
                'coef_u': (['t', 'kx', 'kz'], coef_u),
                'coef_w': (['t', 'kx', 'kz'], coef_w),
            },
            coords={
                't': data.t,
                'kx': range(coef_theta.shape[1]),
                'kz': range(coef_theta.shape[2]),
            }
        )

        data = xr.merge([data, coef_data])

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
    argParser.add_argument(
        '--coef', action='store_true', help='Include coefficient data')
    args = argParser.parse_args()

    files = glob.glob(os.path.join(args.dir, '*.h5'))
    for i, file in enumerate(files):
        logger.info(f'[{i+1:d}/{len(files):d}]\tProcessing {file:s}')
        outfile = post_process(file, args.coef)
        logger.info(f'[{i+1:d}/{len(files):d}]\tSaved {outfile:s}')
