#!/usr/bin/env python3

# fits2img.py, 2016, (c) Anton Lukyanov

"""
Converts images in .fits format into one of image formats supported by scipy.misc.imsave.
PyFITS is required.
"""

import re
import argparse
import os
import os.path as path
import pyfits
from scipy.misc import imsave


def convert(source, destination):
    with pyfits.open(source) as f:
        # Count the number of actual images.
        images_num = 0
        for hdu in f:
            if hdu.is_image and hdu.header['naxis'] > 0:
                images_num += 1
        for i, hdu in enumerate(f):
            if hdu.is_image and hdu.header['naxis'] > 0:
                hdu.verify('silentfix')
                if images_num > 1:
                    destination = re.sub('(\.[^\.]+)$', r'__%03d\1' % i, destination)
                imsave(destination, hdu.data)
                break


parser = argparse.ArgumentParser(
    description='Converts images in fits format into one of image formats supported by scipy.misc.imsave.')
parser.add_argument('source',
    help='Source image in .fits format or a directory with images. In case of directory converts '
         'all images inside it.',
    action='store')
parser.add_argument('destination',
    nargs='?',
    help='If destination is a directory, saves output inside that directory. Otherwise it saves '
         'source image with specified name and format. If not specified, saves images in png '
         'format with the same names in the same folder.',
    action='store')
parser.add_argument('-r', '--recursively',
    help='Process all subdirectories.',
    action='store_true')
parser.add_argument('-f', '--format',
    help='Saves images in specified format. Use when destination is a directory.',
    action='store')

args = parser.parse_args()

dirs = []
if path.isdir(args.source):
    dirs.append(args.source)
    # Collecting all subdirectories recursively.
    if args.recursively:
        dirs_to_proc = []
        dirpath = args.source
        while dirpath:
            for e in os.scandir(dirpath):
                if e.is_dir():
                    dirs.append(e.path)
                    dirs_to_proc.append(e.path)
            try:
                dirpath = dirs_to_proc.pop()
            except IndexError:
                dirpath = False

ext = '.' + ('png' if not args.format else args.format)

if dirs:
    if args.destination:
        dest_dir = args.destination
    else:
        dest_dir = dirs[0]
    for d in dirs:
        dest_filedir = path.join(dest_dir, '/'.join(d.split('/')[1:]))
        if not path.isdir(dest_filedir):
            os.makedirs(dest_filedir)
        for e in os.scandir(d):
            if e.is_file() and not e.is_symlink() and e.path.endswith('.fits'):
                dest_filepath = path.join(
                    dest_filedir,
                    path.basename(re.sub('\.[^\.]+$', ext, e.path))
                )
                convert(e.path, dest_filepath)
else:
    if not args.destination:
        dest = re.sub('\.[^\.]+$', ext, args.source)
    else:
        dest = args.destination
    convert(args.source, dest)
