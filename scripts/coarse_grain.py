"""Script for filtering fine model tendencies."""

import argparse
import logging

import dedalus.public as d3
from mpi4py import MPI
import numpy as np
import xarray as xr
import yaml

from modules import coarse_graining

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Coarse-grains model output.'
    )
    parser.add_argument('config', help='YAML config file')
    parser.add_argument('-k', '--key', help='Top-level config key')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.UnsafeLoader)

    if args.key is not None:
        config = config[args.key]

    comm.Barrier()
    logger = logging.getLogger()
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.CRITICAL + 1)  # disable logging entirely

    # Dedalus adds a StreamHandler to the root logger so we need to remove it
    for h in logging.root.handlers:
        logging.root.removeHandler(h)

    handler = logging.FileHandler(config['logfile'])
    formatter = logging.Formatter(
        '%(asctime)s %(name)s {}/{} %(levelname)s :: %(message)s'
        .format(rank, size)
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info(' Configuration ')
    logger.info('===============\n' + yaml.dump(config).rstrip('\n'))

    input_data = xr.open_mfdataset(config['input'])
    to_nx = config['parameters']['to_nx']
    to_nz = config['parameters']['to_nz']
    u_out = np.zeros((input_data.t.size, to_nx, to_nz))
    w_out = np.zeros((input_data.t.size, to_nx, to_nz))
    theta_out = np.zeros((input_data.t.size, to_nx, to_nz))
    coarse_grainer = coarse_graining.CoarseGrainer(
        config['parameters']['aspect'], input_data.x.size, input_data.z.size,
    )

    for i in range(input_data.t.size):
        if i % 10 == 0:
            logger.info(f'Processing snapshot {i+1} of {input_data.t.size}')

        snapshot = input_data.isel(t=i).compute()
        u_out[i,...], w_out[i,...], theta_out[i,...] = coarse_grainer.run(
            snapshot.u.data, snapshot.w.data, snapshot.theta.data,
            time=config['parameters']['run_time'],
            dt=config['parameters']['run_dt'],
            to_shape=(to_nx, to_nz),
        )

    logger.info('Finishing...')
    coords = d3.CartesianCoordinates('x', 'z')
    dist = d3.Distributor(coords, dtype=np.float64)
    xbasis = d3.RealFourier(
        coords['x'], size=to_nx, bounds=(0, config['parameters']['aspect']))
    zbasis = d3.ChebyshevT(coords['z'], size=to_nz, bounds=(0, 1))
    x_out = xbasis.global_grid().squeeze()
    z_out = zbasis.global_grid().squeeze()

    out = xr.Dataset(
        data_vars={
            'u': (('t', 'x', 'z'), u_out),
            'w': (('t', 'x', 'z'), w_out),
            'theta': (('t', 'x', 'z'), theta_out),
        },
        coords={'t': input_data.t, 'x': x_out, 'z': z_out}
    )
    if rank == 0:
        out.to_netcdf(config['output'])
    logger.info(f"Data saved to {config['output']}")
