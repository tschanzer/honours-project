"""Script for simulating RBC with nonlinear hyperdiffusion."""

import argparse
import logging

from modules import models

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # Parse command line arguments
    argParser = argparse.ArgumentParser(
        description=('Runs a direct numerical simulation of RBC.')
    )
    argParser.add_argument(
        '--aspect', type=float, help='Aspect ratio', required=True)
    argParser.add_argument(
        '--Nx', type=int, help='# horizontal modes', required=True)
    argParser.add_argument(
        '--Nz', type=int, help='# vertical modes', required=True)
    argParser.add_argument(
        '--Ra', type=float, help='Rayleigh number', required=True)
    argParser.add_argument(
        '--Pr', type=float, help='Prandtl number', required=True)
    argParser.add_argument(
        '--hyper', type=float, help='Hyperdiffusivity', required=True)
    argParser.add_argument(
        '--time', type=float, help='Simulation time', required=True)
    argParser.add_argument(
        '--dt', type=float, help='Time step', required=True)
    argParser.add_argument(
        '--out', type=str, help='Output directory', required=True)
    argParser.add_argument(
        '--save', type=int, help='Output interval (# steps)', required=True)
    argParser.add_argument(
        '--max-writes', type=int, help='Max writes per file', default=1000)
    argParser.add_argument(
        '--restart-file', help='Restart file (optional)', default=None)
    argParser.add_argument(
        '--load-file', default=None,
        help='Existing data file or glob pattern for loading state')
    argParser.add_argument(
        '--load-time', help='Sim time of initial condition',
        type=int, default=None)
    args = argParser.parse_args()

    if args.restart_file:
        logger.info('')
        logger.info('RESTART RUN')
        logger.info('===========')
    else:
        logger.info('Nonlinear Hyperdiffusion Model')
        logger.info('==============================')

    model = models.NonlinearHyperdiffusionModel(
        aspect=args.aspect, Nx=args.Nx, Nz=args.Nz, Rayleigh=args.Ra,
        Prandtl=args.Pr, hyper=args.hyper,
    )

    if args.restart_file:
        model.restart(args.restart_file)
    elif args.load_file:
        model.load_from_existing(args.load_file, args.load_time)
    else:
        model.set_initial_conditions()

    model.log_data(args.out, args.save, args.max_writes)
    model.run(args.time, args.dt)
