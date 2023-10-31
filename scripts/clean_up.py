"""Utility for deleting .h5 files after conversion to netCDF."""

import argparse
import glob
import os
import shutil


def clean_up(path):
    files = glob.glob(f'{path}/**/*_s*.nc', recursive=True)
    dirnames = [os.path.dirname(f) for f in files]
    dirnames = sorted(list(set(dirnames)))

    for d in dirnames:
        nc_files = glob.glob(os.path.join(d, '*_s*.nc'))
        h5files = []
        p_dirs = []
        for f in nc_files:
            h5file = f.removesuffix('nc') + 'h5'
            if os.path.exists(h5file): h5files.append(h5file)

            p_dir = f.removesuffix('.nc')
            if os.path.exists(p_dir): p_dirs.append(p_dir)

        if len(h5files) == 0 and len(p_dirs) == 0: continue

        print(d)
        print('-'*len(d))
        print('.h5 files:')
        for f in h5files: print('\t' + os.path.basename(f))
        print('Directories:')
        for p_dir in p_dirs: print('\t' + os.path.basename(p_dir))
        print('')
        confirm = input('Confirm deletion? (Y/n)')
        if confirm == 'Y':
            for f in h5files: os.remove(f)
            for p_dir in p_dirs: shutil.rmtree(p_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Deletes h5 files that have been converted to netCDF.'
    )
    parser.add_argument('path', help='Path to search for h5 files')
    args = parser.parse_args()
    clean_up(args.path)
