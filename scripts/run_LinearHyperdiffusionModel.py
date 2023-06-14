"""Script for simulating RBC with linear hyperdiffusion."""

import argparse
import logging

import modules.models as models

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
        '--restart-file', help='Restart file (optional)', default=None)
    args = argParser.parse_args()

    if args.restart_file:
        logger.info('')
        logger.info('RESTART RUN')
        logger.info('===========')
    else:
        logger.info('Linear Hyperdiffusion Model')
        logger.info('===========================')

    model = models.LinearHyperdiffusionModel(
        aspect=args.aspect, Nx=args.Nx, Nz=args.Nz, Rayleigh=args.Rayleigh,
        Prandtl=args.Prandtl, hyper_coef=args.hyper,
    )

    if args.restart_file:
        model.restart(args.restart_file)
    else:
        model.set_initial_conditions()

    model.log_data(args.out, args.save)
    model.run(args.time, args.dt)