"""Script for running Rayleigh-Bénard models."""

import argparse
import logging
import os

from mpi4py import MPI
import yaml

from modules import models

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


def run(conf):
    """
    Run a simulation using a configuration dictionary.

    Args:
        config: Configuration dictionary.
    """

    comm.Barrier()
    handler = logging.FileHandler(conf['logfile'])
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if conf['initial_condition']['type'] == 'restart':
        logger.info('========================')
        logger.info('| Simulation restarted |')
        logger.info('========================')
    logger.info(' Configuration ')
    logger.info('===============\n%s', yaml.dump(conf).rstrip('\n'))

    equations = conf['equations'].pop('class')
    model = models.Model(
        equations=equations, **conf['parameters'], **conf['equations'],
    )

    match conf['initial_condition'].pop('type'):
        case 'prescribed':
            model.set_initial_conditions()
        case 'restart':
            model.restart(**conf['initial_condition'])
        case 'interpolate':
            model.interp_from_existing(**conf['initial_condition'])
        case _:
            raise ValueError('Invalid initial condition type')

    model.log_data(**conf['output'], timestep=conf['run']['timestep'])
    if 'pair_output' in conf:
        model.log_data_plusdt(**conf['pair_output'])

    model.run(**conf['run'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Runs Rayleigh-Bénard models'
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
        if config['initial_condition']['type'] != 'restart':
            os.mkdir(config['output']['dir'])

    run(config)
