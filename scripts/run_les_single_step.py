"""Script for large eddy simulation of Rayleigh-Benard convection."""

import argparse
import logging

from modules import models

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # Parse command line arguments
    argParser = argparse.ArgumentParser(
        description=('Runs a large eddy simulation of RBC.')
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
        '--filter-std', type=float,
        help='Filter standard deviation', required=True)
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

    logger.info('Single step coarse LES model run')
    logger.info('============================')

    model = models.LESSingleStepModel(
        aspect=args.aspect, Nx=args.Nx, Nz=args.Nz, Rayleigh=args.Ra,
        Prandtl=args.Pr,
    )

    model.input_highres(args.highres)
    model.log_data_t(args.out_t, args.max_writes)
    model.log_data_tplusdt(args.out_tplusdt)

    model.run(args.dt, args.filter_std)
