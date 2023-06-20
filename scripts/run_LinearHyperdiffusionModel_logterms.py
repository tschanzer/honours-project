"""Script for simulating RBC with linear hyperdiffusion."""

import argparse
import logging
import os

import dedalus.public as d3

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
    args = argParser.parse_args()

    logger.info('Linear Hyperdiffusion Model')
    logger.info('===========================')

    model = models.LinearHyperdiffusionModel(
        aspect=args.aspect, Nx=args.Nx, Nz=args.Nz, Rayleigh=args.Ra,
        Prandtl=args.Pr, hyper_coef=args.hyper,
    )

    model.set_initial_conditions()
    model.log_data(args.out, args.save)

    terms = model.solver.evaluator.add_file_handler(
        os.path.join(args.out, 'terms'), iter=1000, max_writes=1000)
    terms.add_tasks(
        [
            model.Prandtl/model.Rayleigh*d3.lap(model.fields['u']),
            (-model.Prandtl/model.Rayleigh*model.hyper_coef*model.taper
             *d3.lap(d3.lap(model.fields['u']))
             ),
            -model.Prandtl/model.Rayleigh*d3.grad(model.fields['pi']),
            -model.fields['u']@d3.grad(model.fields['u']),
            1/model.Rayleigh*d3.lap(model.fields['theta']),
            (-1/model.Rayleigh*model.hyper_coef*model.taper
             *d3.lap(d3.lap(model.fields['theta']))
             ),
            -model.fields['u']@d3.grad(model.fields['theta']),
        ],
        layout='g',
    )

    model.run(args.time, args.dt)
