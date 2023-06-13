"""Script to make an animation of the temperature and vorticity data."""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import xarray as xr
import os
import argparse

plt.rcParams.update({
    'animation.embed_limit': 1000,
    'animation.ffmpeg_path': (
        '/srv/ccrc/AtmSS/z5310829/miniconda3/envs/hons/bin/ffmpeg'
    )
})


def animate(file, framerate=30, timesteps_per_frame=1):
    """
    Makes an animation of the temperature and vorticity data.

    Args:
        file: Path to netCDF data file.
        framerate: Animation frame rate.
        timesteps_per_frame: Set to n to show only every nth timestep.
    """

    data = xr.open_dataset(file)
    data['omega'] = data.u.differentiate('z') - data.w.differentiate('x')
    omega_lim = 2*data.omega.std()

    fig, axes = plt.subplots(2, 1, figsize=(11, 3.5), sharex=True)

    mesh1 = axes[0].pcolormesh(
        data.x, data.z, data.theta.isel(t=0).T, cmap='RdBu_r')
    mesh1.set_clim(-0.5, 0.5)
    fig.colorbar(mesh1, ax=axes[0], label='$\\theta$', pad=0.02, shrink=0.75)
    axes[0].set(ylabel='$z$')
    axes[0].set_aspect('equal')
    title = axes[0].set_title('')

    mesh2 = axes[1].pcolormesh(
        data.x, data.z, data.omega.isel(t=0).T, cmap='RdBu_r')
    mesh2.set_clim(-omega_lim, omega_lim)
    fig.colorbar(mesh2, ax=axes[1], label='$\\omega_y$', pad=0.02, shrink=0.75)
    axes[1].set(xlabel='x', ylabel='$z$')
    axes[1].set_aspect('equal')
    fig.tight_layout()

    def update(i):
        mesh1.set_array(data.theta.isel(t=i).T)
        mesh2.set_array(data.omega.isel(t=i).T)
        title.set_text(f'$t$ = {data.t[i]*1e-6:.2f} $\\times 10^6$')
        return mesh1, mesh2, title

    frames = range(0, data.t.size, timesteps_per_frame)
    ani = FuncAnimation(
        fig, update, frames=frames, blit=True, interval=1e3/framerate)
    plt.close()

    outdir = os.path.dirname(file)
    filename = os.path.basename(file).removesuffix('.nc') + '.mp4'
    outfile = os.path.join(outdir, filename)
    ani.save(outfile)


if __name__ == '__main__':
    argParser = argparse.ArgumentParser(
        description=(
            'Makes an animation of the temperature and vorticity data')
    )
    argParser.add_argument('file', type=str, help='Path to data file')
    argParser.add_argument(
        '--framerate', type=float, default=30, help='Frame rate')
    argParser.add_argument(
        '--steps', type=int, default=1,
        help='Number of time steps per frame')
    args = argParser.parse_args()

    animate(args.file, args.framerate, args.steps)
