import re
import os
from collections import OrderedDict
from datetime import datetime
from typing import List, Union, Dict, Tuple
from os import path

import numpy as np


class Timer:
    """
    Простой таймер для замеров времени выполнения кода.
    """

    def __init__(self):
        self.start = datetime.now()

    def show(self):
        print('Running time: %s' % (datetime.now() - self.start))


def find_images(dirpath: str) -> Tuple[List[str], int]:
    imagepaths = []
    for e in os.scandir(dirpath + '_png'):
        if e.is_file() and not e.is_symlink() and e.path.endswith('.png'):
            imagepaths.append(e.path)
    images_num = len(imagepaths)
    print('Found images: %d' % images_num)
    return imagepaths, images_num


def parse_flare_class(value: float) -> Union[str, None]:
    """
    Accepts GOES FLUX value and returns flare class B, C, M or X.
    """
    if 1 <= value < 10:
        letter = 'B'
        coef = 1
    elif 10 <= value < 100:
        letter = 'C'
        coef = 10
    elif 100 <= value < 1000:
        letter = 'M'
        coef = 100
    elif value >= 1000:
        letter = 'X'
        coef = 1000
    else:
        return None
    return '%s%.1f' % (letter, value / coef)


Flares = Dict[str, Tuple[str, int, int]]


def parse_flares(filepath: str) -> Flares:
    """
    Parses file with flares description and return ordered dictionary which maps dates to tuples
    (flare class, hours, minutes). Function expects lines in the following form:

        20110218 10:11 661

    :param filepath:
    :return:
    """

    flares = OrderedDict()
    if path.isfile(filepath):
        with open(filepath) as f:
            for ln in f:
                match = re.match(
                    '^([0-9]{4})([0-9]{2})([0-9]{2})\s'
                    '([0-9]{1,2}):([0-9]{1,2})\s'
                    '([0-9]+\.?[0-9]*)$',
                    ln
                )
                if match:
                    y, m, d = match.group(1), match.group(2), match.group(3)
                    hours = int(match.group(4))
                    minutes = int(match.group(5))
                    value = float(match.group(6))
                    key = '-'.join([y, m, d])
                    if key not in flares:
                        flares[key] = []
                    flares[key].append((parse_flare_class(value), hours, minutes))
    return flares


def parse_dates(filepaths: List[str]) -> Dict[str, List[int]]:
    dates = {}
    for i, p in enumerate(filepaths):
        date = re.match('^.+720s\.([0-9]{4})([0-9]{2})([0-9]{2})_.+$', p)
        if date:
            y = date.group(1)
            m = date.group(2)
            d = date.group(3)
            date = '%s-%s-%s' % (y, m, d)
            if date not in dates:
                dates[date] = []
            dates[date].append(i)
    return dates


def moving_average_smoothing(vals: np.ndarray, n: int) -> np.ndarray:
    vals = np.asarray(vals)
    average = vals[:n].sum() / n
    vals_ma = vals.copy()
    vals_ma[n - 1] = average
    for i in range(n, len(vals)):
        average -= (vals[i - n] - vals[i]) / n
        vals_ma[i] = average
    return vals_ma


def laplacian_smoothing(vals: np.ndarray, itern: int = 1) -> np.ndarray:
    if not isinstance(vals, np.ndarray):
        vals = np.asarray(vals)

    for _ in range(itern):
        for i in range(1, len(vals) - 1):
            vals[i] = 0.25 * vals[i-1] + 0.5 * vals[i] + 0.25 * vals[i+1]

    return vals


def apply_smoothing(
    vals: np.ndarray, lap_smoothing: bool = False, lap_smoothing_itern: int = 1,
    sma_smoothing: bool = False, sma_smoothing_wnd: int = 2
) -> np.ndarray:
    # Simple moving average smoothing.
    if sma_smoothing:
        vals = moving_average_smoothing(vals, sma_smoothing_wnd)
    # Laplacian smoothing.
    if lap_smoothing:
        vals = laplacian_smoothing(vals, lap_smoothing_itern)
    return vals
