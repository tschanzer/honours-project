"""Script for running Rayleigh-Bénard resolution tests."""

import argparse
import copy
import os

import yaml
from mpi4py import MPI

import run_model

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Runs Rayleigh-Bénard resolution tests'
    )
    parser.add_argument('config', help='YAML config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.UnsafeLoader)

    base_config = config['base']
    resolutions = config['runs']

    base_dir = os.path.dirname(base_config['output']['dir'])
    base_max_writes = base_config['output']['max_writes']
    base_nx = base_config['parameters']['nx']
    base_nz = base_config['parameters']['nz']
    base_dt = base_config['run']['timestep']
    aspect = base_config['parameters']['aspect']

    for new_nx, new_nz in resolutions:
        new_config = copy.deepcopy(base_config)

        # Update resolution
        new_config['parameters']['nx'] = new_nx
        new_config['parameters']['nz'] = new_nz

        # Update output directory and logfile
        name = f'{new_nx}x{new_nz}'
        out_dir = os.path.join(base_dir, name)
        new_config['output']['dir'] = out_dir
        new_config['logfile'] = os.path.join(out_dir, name + '.log')

        # Number of writes per file scales inversely with new_nx*new_nz
        new_config['output']['max_writes'] = round(
            base_max_writes*(base_nx/new_nx)*(base_nz/new_nz)
        )

        # Time step scales to approximately preserve Courant number
        new_config['run']['timestep'] = base_dt*(
            (base_nx/aspect + base_nz)/(new_nx/aspect + new_nz)
        )

        # Create the output directory so the logfile can be created.
        # If the directory already exists, skip this resolution.
        if rank == 0 and base_config['initial_condition'] != 'restart':
            try:
                os.mkdir(new_config['output']['dir'])
            except FileExistsError:
                continue

        run_model.run(new_config)
