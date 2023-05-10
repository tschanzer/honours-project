"""Script to run direct numerical simulations of RBC."""

import rbc_setup
import logging
import argparse
logger = logging.getLogger(__name__)


def add_file_handler(solver, data_dir, save_interval):
    """
    Tells the solver to save output to a file.

    Args:
        solver: Dedalus Solver object.
        data_dir: Directory for data output.
        save_interval: Number of time steps between saves.
    """

    snapshots = solver.evaluator.add_file_handler(
        data_dir, iter=save_interval, max_writes=1000)
    snapshots.add_tasks(solver.state, layout='g')


def run_dns(solver, timestep):
    """
    Runs a direct numerical simulation of RBC.

    Args:
        solver: Dedalus Solver object.
        timestep: Time step.
    """

    try:
        logger.info('Starting main loop.')
        while solver.proceed:
            if solver.iteration % 1000 == 0:
                logger.info(
                    f'Iteration {solver.iteration:d}\t\t'
                    f'Sim time {solver.sim_time:.3e}'
                )
            solver.step(timestep)
    except:
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
        '--Lx', type=float, help='Domain length', required=True,
    )
    argParser.add_argument(
        '--Nx', type=int, help='# horizontal modes', required=True,
    )
    argParser.add_argument(
        '--Nz', type=int, help='# vertical modes', required=True,
    )
    argParser.add_argument(
        '--Ra', type=float, help='Rayleigh number', required=True,
    )
    argParser.add_argument(
        '--Pr', type=float, help='Prandtl number', required=True,
    )
    argParser.add_argument(
        '--time', type=float, help='Simulation time', required=True,
    )
    argParser.add_argument(
        '--dt', type=float, help='Timestep in units of 1/Ra', required=True,
    )
    argParser.add_argument(
        '--out', type=str, help='Output directory', required=True,
    )
    argParser.add_argument(
        '--save', type=int, help='Output interval (# steps)', required=True,
    )
    args = argParser.parse_args()

    logger.info('Rayleigh-Bénard convection: direct numerical simulation')

    solver = rbc_setup.build_solver(
        args.Lx, args.Nx, args.Nz, args.Ra, args.Pr)
    solver.stop_sim_time = args.time
    rbc_setup.set_initial_conditions(
        solver, 'random_theta', sigma=1e-2*args.Ra)
    add_file_handler(solver, args.out, args.save)
    logger.info(f'Saving data to {args.out:s} every {args.save:d} timesteps.')

    logger.info(
        f'Running for {args.time:.3g} units of simulation time '
        f'with dt = {args.dt/args.Ra:.3e}')
    run_dns(solver, args.dt/args.Ra)
