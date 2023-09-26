"""Script for filtering fine model tendencies."""

import argparse
import logging

import dedalus.public as d3
from mpi4py import MPI
import numpy as np
import xarray as xr

from modules import models

logger = logging.getLogger(__name__)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def tendency(state_t, state_tplusdt):
    tend = (state_tplusdt.drop('t') - state_t.drop('t'))/state_t.timestep
    return tend.assign_coords({'t': state_t.t})


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description=('Filters fine tendencies.'))
    parser.add_argument(
        '--aspect', type=float, help='Aspect ratio', required=True)
    parser.add_argument(
        '--to-Nx', help='# horizontal modes in output',
        type=int, required=True)
    parser.add_argument(
        '--to-Nz', help='# vertical modes in output',
        type=int, required=True)
    parser.add_argument(
        '--time', type=float, help='Diffusion time', required=True)
    parser.add_argument(
        '--dt', type=float, help='Diffusion time step', required=True)
    parser.add_argument(
        '--data-t', help='Path to time t source data', required=True)
    parser.add_argument(
        '--data-tplusdt', help='Path to time t+dt source data', required=True)
    parser.add_argument(
        '--out', help='Output file', required=True)
    args = parser.parse_args()

    data_t = xr.open_mfdataset(args.data_t)
    data_tplusdt = xr.open_mfdataset(args.data_tplusdt)
    u_out = np.zeros((data_tplusdt.t.size, args.to_Nx, args.to_Nz))
    w_out = np.zeros((data_tplusdt.t.size, args.to_Nx, args.to_Nz))
    theta_out = np.zeros((data_tplusdt.t.size, args.to_Nx, args.to_Nz))
    model = models.TendencyDiffusionModel(
        args.aspect, data_t.x.size, data_t.z.size)

    for i in range(data_tplusdt.t.size):
        if i % 10 == 0:
            logger.info(f'Processing snapshot {i+1} of {data_tplusdt.t.size}')

        tend = tendency(data_t.isel(t=i), data_tplusdt.isel(t=i)).compute()
        u_out[i,...], w_out[i,...], theta_out[i,...] = model.run(
            tend.u.data, tend.w.data, tend.theta.data,
            time=args.time, dt=args.dt, to_shape=(args.to_Nx, args.to_Nz),
        )

    logger.info('Finishing...')
    coords = d3.CartesianCoordinates('x', 'z')
    dist = d3.Distributor(coords, dtype=np.float64)
    xbasis = d3.RealFourier(
        coords['x'], size=args.to_Nx, bounds=(0, args.aspect))
    zbasis = d3.ChebyshevT(coords['z'], size=args.to_Nz, bounds=(0, 1))
    x_out = xbasis.global_grid().squeeze()
    z_out = zbasis.global_grid().squeeze()

    out = xr.Dataset(
        data_vars={
            'u': (('t', 'x', 'z'), u_out),
            'w': (('t', 'x', 'z'), w_out),
            'theta': (('t', 'x', 'z'), theta_out),
        },
        coords={'t': data_t.t[:data_tplusdt.t.size], 'x': x_out, 'z': z_out}
    )
    if rank == 0:
        out.to_netcdf(args.out)
    logger.info(f'Data saved to {args.out}')
