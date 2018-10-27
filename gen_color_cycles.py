#!/usr/bin/env python3

"""
Copyright (c) 2018 Matthew Petroff

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import argparse
import random
import time
import itertools
import platform
import numpy as np
import numba
import joblib
import color_conversions


#
# Configuration
#

parser = argparse.ArgumentParser(
    description="Generate color sets with minimum perceptual distance requirement.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--min-color-dist",
    default=20,
    type=int,
    help="Minimum perceptual color distance (including for CVD)",
)
parser.add_argument(
    "--min-light-dist",
    default=2,
    type=int,
    help="Minimum lightness distance (for grayscale conversion)",
)
parser.add_argument("--num-colors", default=8, type=int, help="Number of colors in set")
parser.add_argument(
    "--cvd-severity",
    default=75,
    type=int,
    help="Severity percentage for CVD simulation",
)
parser.add_argument("--min-j", default=40, type=float, help="Minimum color lightness")
parser.add_argument("--max-j", default=90, type=float, help="Maximum color lightness")
parser.add_argument(
    "--num-sets", default=10000, type=int, help="Number of sets to generate"
)
parser.add_argument(
    "--num-jobs", default=28, type=int, help="Number of parallel jobs to use"
)
args = parser.parse_args()

MIN_COLOR_DIST = args.min_color_dist
MIN_LIGHT_DIST = args.min_light_dist
NUM_COLORS = args.num_colors
CVD_SEVERITY = args.cvd_severity
MIN_J = args.min_j
MAX_J = args.max_j
NUM_SETS = args.num_sets
NUM_JOBS = args.num_jobs

OUT_FILE = (
    f"colors_mcd{MIN_COLOR_DIST}_mld{MIN_LIGHT_DIST}_nc{NUM_COLORS}"
    + f"_cvd{CVD_SEVERITY}_minj{MIN_J}_maxj{MAX_J}_ns{NUM_SETS}"
)


#
# Generate list of colors
#

# Since CAM02-UCS conversions are computationally expensive, but the
# 16.8 million possible 8-bit RGB colors easily fit in memory, we precompute
# the list of colors for normal color vision and three types of color vision
# deficiency. Additionally, very dark and very light colors are discarded,
# leaving 13.1 million colors ($J \in [40, 90]$).


@numba.njit
def calc_jab_colors():
    """
    Calculates CAM02-UCS colors for all 8-bit RGB colors with $J \in [40, 90]$.
    """
    rgb_colors = np.empty((256 ** 3, 3), dtype=np.uint8)
    jab_colors = np.empty((256 ** 3, 3), dtype=np.float32)
    deut_jab_colors = jab_colors.copy()
    prot_jab_colors = jab_colors.copy()
    trit_jab_colors = jab_colors.copy()
    c = 0
    for i in range(256 ** 3):
        r = i % 256
        g = (i // 256) % 256
        b = i // (256 ** 2)
        rgb_linear = color_conversions.sRGB1_to_sRGB1_linear(
            np.array((r / 255, g / 255, b / 255))
        )
        jab = color_conversions.rgb_linear_to_jab(rgb_linear)
        if jab[0] >= MIN_J and jab[0] <= MAX_J:
            rgb_colors[c] = np.array((r, g, b))
            jab_colors[c] = jab
            deut_jab_colors[c] = color_conversions.rgb_linear_to_jab(
                color_conversions.CVD_forward_deuteranomaly(rgb_linear, CVD_SEVERITY)
            )
            prot_jab_colors[c] = color_conversions.rgb_linear_to_jab(
                color_conversions.CVD_forward_protanomaly(rgb_linear, CVD_SEVERITY)
            )
            trit_jab_colors[c] = color_conversions.rgb_linear_to_jab(
                color_conversions.CVD_forward_tritanomaly(rgb_linear, CVD_SEVERITY)
            )
            c += 1
    rgb_colors = rgb_colors[:c]
    jab_colors = jab_colors[:c]
    deut_jab_colors = deut_jab_colors[:c]
    prot_jab_colors = prot_jab_colors[:c]
    trit_jab_colors = trit_jab_colors[:c]
    return rgb_colors, jab_colors, deut_jab_colors, prot_jab_colors, trit_jab_colors


t = time.time()
RGB_COLORS, JAB_COLORS, DEUT_JAB_COLORS, PROT_JAB_COLORS, TRIT_JAB_COLORS = (
    calc_jab_colors()
)
print(f"Color list generated in {time.time() - t}s")


#
# Generate color set
#

# To generate a color set, a starting color is chosen at random. Then, each
# possible color is checked to see if it is far enough away in both lightness
# and perceptual distance, both for normal color vision and for those with
# color vision deficiency. Of these remaining colors, one is chosen at random.
# The process is then repeated until the color set contains the desired number
# of colors. This method has an advantage over rejection sampling, since it is
# guaranteed to return. Checking a coarse CVD interval during set generation
# was tried but removed, since the performance penalty outweighs the gains from
# having to try again fewer times.

COMBINATIONS = tuple(itertools.combinations(range(NUM_COLORS), 2))


@numba.njit
def gen_color_set(seed):
    """
    Generates color set using specified PRNG seed.
    """
    np.random.seed(seed)
    jab_colors = np.empty((NUM_COLORS, 3), dtype=np.float32)
    deut_jab_colors = jab_colors.copy()
    prot_jab_colors = jab_colors.copy()
    trit_jab_colors = jab_colors.copy()
    rgb_colors = np.empty((NUM_COLORS, 3), dtype=np.uint8)
    first_color_idx = np.random.randint(0, RGB_COLORS.shape[0])
    rgb_colors[0] = RGB_COLORS[first_color_idx]
    jab_colors[0] = JAB_COLORS[first_color_idx]
    deut_jab_colors[0] = DEUT_JAB_COLORS[first_color_idx]
    prot_jab_colors[0] = PROT_JAB_COLORS[first_color_idx]
    trit_jab_colors[0] = TRIT_JAB_COLORS[first_color_idx]
    valid_colors = np.empty(RGB_COLORS.shape[0], dtype=np.uint32)
    for i in range(1, NUM_COLORS):
        c = 0
        for j in range(RGB_COLORS.shape[0]):
            for k in range(i):
                if (
                    np.abs(JAB_COLORS[j][0] - jab_colors[k][0]) < MIN_LIGHT_DIST
                    or color_conversions.cam02de(JAB_COLORS[j], jab_colors[k])
                    < MIN_COLOR_DIST
                    or color_conversions.cam02de(DEUT_JAB_COLORS[j], deut_jab_colors[k])
                    < MIN_COLOR_DIST
                    or color_conversions.cam02de(PROT_JAB_COLORS[j], prot_jab_colors[k])
                    < MIN_COLOR_DIST
                    or color_conversions.cam02de(TRIT_JAB_COLORS[j], trit_jab_colors[k])
                    < MIN_COLOR_DIST
                ):
                    break
            else:
                valid_colors[c] = j
                c += 1
        if c == 0:
            return None
        pick = np.random.randint(0, c)
        rgb_colors[i] = RGB_COLORS[valid_colors[pick]]
        jab_colors[i] = JAB_COLORS[valid_colors[pick]]
        deut_jab_colors[i] = DEUT_JAB_COLORS[valid_colors[pick]]
        prot_jab_colors[i] = PROT_JAB_COLORS[valid_colors[pick]]
        trit_jab_colors[i] = TRIT_JAB_COLORS[valid_colors[pick]]

    return rgb_colors


@numba.njit
def check_color_set(rgb_colors):
    """
    Check at finer CVD simulation interval.
    Returns True if colors set is okay, else False
    """
    min_dist = 100
    deut_jab_test = np.empty((NUM_COLORS, 3), dtype=np.float32)
    prot_jab_test = deut_jab_test.copy()
    trit_jab_test = deut_jab_test.copy()
    for severity in range(1, CVD_SEVERITY):
        for i in range(NUM_COLORS):
            rgb_linear = color_conversions.sRGB1_to_sRGB1_linear(rgb_colors[i] / 255)
            deut_jab_test[i] = color_conversions.rgb_linear_to_jab(
                color_conversions.CVD_forward_deuteranomaly(rgb_linear, severity)
            )
            prot_jab_test[i] = color_conversions.rgb_linear_to_jab(
                color_conversions.CVD_forward_protanomaly(rgb_linear, severity)
            )
            trit_jab_test[i] = color_conversions.rgb_linear_to_jab(
                color_conversions.CVD_forward_tritanomaly(rgb_linear, severity)
            )
        for pair in COMBINATIONS:
            min_dist = min(
                min_dist,
                color_conversions.cam02de(
                    deut_jab_test[pair[0]], deut_jab_test[pair[1]]
                ),
            )
            min_dist = min(
                min_dist,
                color_conversions.cam02de(
                    prot_jab_test[pair[0]], prot_jab_test[pair[1]]
                ),
            )
            min_dist = min(
                min_dist,
                color_conversions.cam02de(
                    trit_jab_test[pair[0]], trit_jab_test[pair[1]]
                ),
            )
        if min_dist < MIN_COLOR_DIST:
            return False
    return True


def sort_colors(colors):
    """
    Sorts colors.
    """
    return colors[np.lexsort(colors[:, ::-1].T)]


def gen_sorted_color_set(seed):
    """
    Generates a sorted color set using specified PRNG seed.
    """
    i = 0
    while True:
        # Keep trying until set generation succeeds
        colors = gen_color_set(seed + i)
        if colors is not None and check_color_set(colors):
            print(
                "CVD check for set generation iteration {} for seed {} failed!".format(
                    i, seed
                )
            )
            break
        print("Set generation iteration {} for seed {} failed!".format(i, seed))
        i += 1
    return sort_colors(colors)


def gen_color_names(colors):
    """
    Convert RGB values into a hexadecimal color string.
    """
    color_names = []
    for color in colors:
        name = "{:02x}{:02x}{:02x}".format(*color)
        color_names.append(name)
    return color_names


np.random.seed(614616785)
num_left = NUM_SETS

t = time.time()
i = 0
results = None
while num_left > 0:
    seeds = np.random.random_integers(2 ** 32, size=num_left)
    new_results = joblib.Parallel(n_jobs=NUM_JOBS)(
        joblib.delayed(gen_sorted_color_set)(i) for i in seeds
    )
    if results is None:
        results = np.unique(np.array(new_results), axis=0)
    else:
        results = np.unique(np.append(results, np.array(new_results)), axis=0)
    num_left = NUM_SETS - results.shape[0]
    i += 1
    print(f"{num_left} set(s) left to generate after {i} iteration(s)")
print(f"{NUM_SETS} color sets generated in {time.time() - t}s using {NUM_JOBS} jobs")

with open(OUT_FILE + ".txt", "w") as out:
    out.write(f"# {OUT_FILE}\n")
    out.write("# Python " + platform.sys.version.replace("\n", "") + "\n")
    out.write(
        f"# NumPy {np.__version__}, Numba {numba.__version__}, Joblib {joblib.__version__}\n"
    )
    for result in results:
        out.write(" ".join(gen_color_names(result)) + "\n")
