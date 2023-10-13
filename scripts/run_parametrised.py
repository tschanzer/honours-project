"""Script for parametrised simulation of Rayleigh-Benard convection."""

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
        '--coef-file', required=True,
        help='Path to parametrisation coefficient csv file.')
    argParser.add_argument(
        '--time', type=float, help='Simulation time', required=True)
    argParser.add_argument(
        '--dt', type=float, help='Time step', required=True)
    argParser.add_argument(
        '--out', help='Output directory', required=True)
    argParser.add_argument(
        '--save', type=int, help='Output interval (# steps)', required=True)
    argParser.add_argument(
        '--max-writes', type=int, help='Max writes per file', default=1000)
    argParser.add_argument(
        '--restart-file', help='Restart file (optional)')
    argParser.add_argument(
        '--load-file',
        help='Existing data file or glob pattern for loading state')
    argParser.add_argument(
        '--load-time', help='Sim time of initial condition', type=int)
    argParser.add_argument(
        '--filter-time', type=float,
        help='Run time of diffusion model for smoothing initial condition')
    argParser.add_argument(
        '--filter-dt', type=float, help='Time step of diffusion model')
    args = argParser.parse_args()

    if args.restart_file:
        logger.info('')
        logger.info('RESTART RUN')
        logger.info('===========')
    else:
        logger.info('Direct numerical simulation')
        logger.info('===========================')

    model = models.ResTendParametrisedModel(
        aspect=args.aspect, Nx=args.Nx, Nz=args.Nz, Rayleigh=args.Ra,
        Prandtl=args.Pr, hyper=args.hyper, coef_file=args.coef_file
    )

    if args.restart_file:
        model.restart(args.restart_file)
    elif args.load_file:
        model.load_from_existing(
            args.load_file, args.load_time, args.filter_time, args.filter_dt)
    else:
        model.set_initial_conditions()

    model.log_data(args.out, args.save, args.max_writes)
    model.run(args.time, args.dt)
