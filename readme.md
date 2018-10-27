# Colorblind-friendly color sets

This repository contains code for randomly generating colorblind-friendly color sets. To generate the color sets, color vision deficiency simulations for various types of deficiencies are performed, and a minimum perceptual difference for the simulated colors is enforced using the CAM02-UCS perceptually uniform color space (where each type of deficiency is treated separately), as is a minimum lightness distance (for grayscale).


## Pregenerated color sets

Pregenerated color sets are included in the `color-sets` directory. The following parameters were used:

```
$ python3 gen_color_cycles.py --num-colors 6 --cvd-severity 100 --min-color-dist 20 --num-sets 10000
10000 color sets generated in 31235.150347471237s using 28 jobs

$ python3 gen_color_cycles.py --num-colors 8 --cvd-severity 100 --min-color-dist 18 --num-sets 10000
10000 color sets generated in 68176.89465022087s using 28 jobs

$ python3 gen_color_cycles.py --num-colors 10 --cvd-severity 100 --min-color-dist 16 --num-sets 10000
10000 color sets generated in 295049.9100484848s using 28 jobs
```


## License

The code contained in this repository is distributed under the MIT License.

The included CVD simulation and color distance calculation implementation is based on [Colorspacious](https://github.com/njsmith/colorspacious), which is [MIT licensed](https://github.com/njsmith/colorspacious/blob/v1.1.0/LICENSE.txt).

The resulting color sets generated are released into the public domain using the [CC0 1.0 Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/).


## Credits

* [Matthew Petroff](https://mpetroff.net/), Original author
* [Colorspacious](https://github.com/njsmith/colorspacious), Basis for CVD simulation and color distance calculations

CVD simulation is based on:

> G. M. Machado, M. M. Oliveira and L. A. F. Fernandes, "A Physiologically-based Model for Simulation of Color Vision Deficiency," in _IEEE Transactions on Visualization and Computer Graphics_, vol. 15, no. 6, pp. 1291-1298, Nov.-Dec. 2009. [doi:10.1109/TVCG.2009.113](https://doi.org/10.1109/TVCG.2009.113)

CIECAM02 and CAM02-UCS overview:

> Luo M.R., Li C. (2013) CIECAM02 and Its Recent Developments. In: Fernandez-Maloigne C. (eds) Advanced Color Image Processing and Analysis. Springer, New York, NY. [doi:10.1007/978-1-4419-6190-7_2](https://doi.org/10.1007/978-1-4419-6190-7_2)
