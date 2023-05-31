"""Script to run direct numerical simulations of RBC."""

import rbc_setup
import logging
import argparse
logger = logging.getLogger(__name__)


def run_dns(solver, timestep):
    """
    Runs a direct numerical simulation of RBC.

    Args:
        solver: Dedalus Solver object.
        timestep: Time step.
    """

    try:
        logger.info(f'Starting main loop with dt = {timestep:.3g}')
        while solver.proceed:
            if solver.iteration % 1000 == 0:
                logger.info(
                    f'Iteration {solver.iteration:d}\t\t'
                    f'Sim time {solver.sim_time:.3e}'
                )
            solver.step(timestep)
    except Exception:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        solver.log_stats()


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
        '--time', type=float, help='Simulation time', required=True)
    argParser.add_argument(
        '--dt', type=float, help='Time step', required=True)
    argParser.add_argument(
        '--out', type=str, help='Output directory', required=True)
    argParser.add_argument(
        '--save', type=int, help='Output interval (# steps)', required=True)
    argParser.add_argument(
        '--coef', action='store_true', help='Store coefficient data')
    argParser.add_argument(
        '--tau', action='store_true', help='Store tau variables')
    argParser.add_argument(
        '--init', choices=['random_theta', 'wavy_theta'],
        required=True, help='Initial condition')
    args = argParser.parse_args()

    logger.info('Rayleigh-Bénard convection: direct numerical simulation')

    solver = rbc_setup.build_solver(
        args.aspect, args.Nx, args.Nz, args.Ra, args.Pr)
    rbc_setup.set_initial_conditions(
        solver, args.init, sigma=1e-2)
    rbc_setup.add_file_handler(
        solver, args.out, args.save, args.coef, args.tau)

    solver.stop_sim_time = args.time
    logger.info(f'Simulation length: t_max = {args.time:.3g}')
    run_dns(solver, args.dt)
