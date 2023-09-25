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
        '--filter-time', type=float,
        help='Run time of diffusion model for smoothing', required=True)
    argParser.add_argument(
        '--filter-dt', type=float,
        help='Time step of diffusion model', required=True)
    argParser.add_argument(
        '--dt', type=float, help='Time step', required=True)
    argParser.add_argument(
        '--out-t', help='Output directory for time t snapshots', required=True)
    argParser.add_argument(
        '--out-tplusdt', required=True,
        help='Output directory for time t+dt snapshots')
    argParser.add_argument(
        '--max-writes', type=int, help='Max writes per file', default=1000)
    argParser.add_argument(
        '--highres', help='High-resolution input data file', required=True)
    args = argParser.parse_args()

    logger.info('Single step coarse model run')
    logger.info('============================')

    model = models.SingleStepModel(
        aspect=args.aspect, Nx=args.Nx, Nz=args.Nz, Rayleigh=args.Ra,
        Prandtl=args.Pr, hyper=args.hyper,
    )

    model.input_highres(args.highres)
    model.init_filter()
    model.log_data_t(args.out_t, args.max_writes)
    model.log_data_tplusdt(args.out_tplusdt)

    model.run(args.dt, args.filter_time, args.filter_dt)
