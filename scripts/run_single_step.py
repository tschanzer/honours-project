"""Script for running Rayleigh-Bénard models."""

import argparse
import logging
import os

from mpi4py import MPI
import yaml

from modules import models

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size


def run(config):
    """
    Run a simulation using a configuration dictionary.

    Args:
        config: Configuration dictionary.
    """

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

    equations = config['equations'].pop('class')
    model = models.SingleStepModel(
        equations=equations, **config['parameters'], **config['equations'],
    )

    model.load_initial_states(config['input'])
    model.log_final_states(
        config['output']['dir'], config['output']['max_writes'],
    )
    model.run(config['run']['dt'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Runs Rayleigh-Bénard models'
    )
    parser.add_argument('config', help='YAML config file')
    parser.add_argument('-k', '--key', help='Top-level config key')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.UnsafeLoader)

    if args.key is not None:
        config = config[args.key]

    # Create the output directory so the logfile can be created
    if rank == 0:
        os.mkdir(config['output']['dir'])

    run(config)
