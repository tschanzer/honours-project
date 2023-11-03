"""Script for filtering fine model tendencies."""

import argparse
import logging
import os

from mpi4py import MPI
import yaml

from modules import coarse_graining

comm = MPI.COMM_WORLD  # pylint: disable=I1101
rank = comm.rank
size = comm.size

logger = logging.root
if rank == 0:
    logger.setLevel(logging.INFO)
else:
    logger.setLevel(logging.CRITICAL + 1)  # disable logging entirely
formatter = logging.Formatter(
    f'%(asctime)s %(name)s {rank}/{size} %(levelname)s :: %(message)s'
)

# Dedalus adds a StreamHandler to the root logger so we need to remove it
for h in logger.handlers:
    logger.removeHandler(h)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Coarse-grains model output.'
    )
    parser.add_argument('config', help='YAML config file')
    parser.add_argument('-k', '--key', help='Top-level config key')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.load(f, yaml.UnsafeLoader)
    if args.key is not None:
        config = config[args.key]

    # Create the output directory so the logfile can be created
    if rank == 0:
        os.mkdir(config['output']['dir'])
    comm.Barrier()
    handler = logging.FileHandler(config['logfile'])
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info(' Configuration ')
    logger.info('===============\n%s', yaml.dump(config).rstrip('\n'))

    coarse_grainer = coarse_graining.CoarseGrainer(
        config['input'], config['parameters']['aspect'],
    )
    to_nx = config['parameters']['to_nx']
    to_nz = config['parameters']['to_nz']
    coarse_grainer.configure_output(
        (to_nx, to_nz), config['output']['dir'],
        config['output']['max_writes'],
    )
    coarse_grainer.run(
        config['parameters']['run_time'], config['parameters']['run_dt'],
    )
