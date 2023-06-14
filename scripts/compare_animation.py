"""Script to make an animation of two runs."""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import xarray as xr
import os
import argparse
import glob

import modules.regridding as regridding

plt.rcParams.update({
    'animation.embed_limit': 1000,
    'animation.ffmpeg_path': (
        '/srv/ccrc/AtmSS/z5310829/miniconda3/envs/hons/bin/ffmpeg'
    )
})


def animate(fine_dir, coarse_dir, framerate=30, timesteps_per_frame=1):
    """
    Makes an animation to compare two runs.

    Args:
        fine_file: Path to fine netCDF data directory.
        coarse_file: Path to coarse netCDF data directory.
        framerate: Animation frame rate.
        timesteps_per_frame: Set to n to show only every nth timestep.
    """

    fine_files = sorted(glob.glob(os.path.join(fine_dir, '*.nc')))
    coarse_files = sorted(glob.glob(os.path.join(coarse_dir, '*.nc')))
    fine_data = xr.open_mfdataset(fine_files)
    coarse_data = xr.open_mfdataset(coarse_files)
    regridder = regridding.Regridder3D(fine_data, coarse_data)
    fine_data = regridder(fine_data.theta.chunk({'t': -1}))

    fig, axes = plt.subplots(2, 1, figsize=(11, 3.5), sharex=True)

    mesh1 = axes[0].pcolormesh(
        fine_data.x, fine_data.z, fine_data.isel(t=0).T, cmap='RdBu_r')
    mesh1.set_clim(-0.5, 0.5)
    fig.colorbar(mesh1, ax=axes[0], label='$\\theta$', pad=0.02, shrink=0.75)
    axes[0].set(ylabel='$z$')
    axes[0].set_aspect('equal')
    title = axes[0].set_title('')

    mesh2 = axes[1].pcolormesh(
        coarse_data.x, coarse_data.z, coarse_data.theta.isel(t=0).T,
        cmap='RdBu_r')
    mesh2.set_clim(-0.5, 0.5)
    fig.colorbar(mesh2, ax=axes[1], label='$\\omega_y$', pad=0.02, shrink=0.75)
    axes[1].set(xlabel='x', ylabel='$z$', title='Coarse model', aspect='equal')
    fig.tight_layout()

    def update(i):
        mesh1.set_array(fine_data.isel(t=i).T)
        mesh2.set_array(coarse_data.theta.isel(t=i).T)
        title.set_text(
            'Coarse-grained fine model\t'
            f'$t$ = {fine_data.t[i]*1e-6:.2f} $\\times 10^6$')
        return mesh1, mesh2, title

    frames = range(0, fine_data.t.size, timesteps_per_frame)
    ani = FuncAnimation(
        fig, update, frames=frames, blit=True, interval=1e3/framerate)
    plt.close()

    filename = fine_files[0].removesuffix('_s1.nc') + '.mp4'
    ani.save(filename)


if __name__ == '__main__':
    argParser = argparse.ArgumentParser(
        description=(
            'Makes an animation of the temperature and vorticity data')
    )
    argParser.add_argument(
        'fine_dor', type=str, help='Path to fine data directory')
    argParser.add_argument(
        'coarse_dor', type=str, help='Path to coarse data directory')
    argParser.add_argument(
        '--framerate', type=float, default=30, help='Frame rate')
    argParser.add_argument(
        '--steps', type=int, default=1,
        help='Number of time steps per frame')
    args = argParser.parse_args()

    animate(args.fine_dor, args.coarse_dor, args.framerate, args.steps)
