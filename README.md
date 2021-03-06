# SCOTT: Shape-location COnsiderate Tracking with OT

This open source Python library provides a cell tracking solver based on Optimal Transport.

Please see our paper for technical details:

```
@article{zheng2020scott,
  title={SCOTT: Shape-Location Combined Tracking with Optimal Transport},
  author={Zheng, Xinye and Ye, Jianbo and Wang, James Z and Li, Jia},
  journal={SIAM Journal on Mathematics of Data Science},
  year={2020}
}
```

## Installation

The library has been tested on Unix system with its mixture-ordered OT solver on Python 3.6+. To install Python package requirements:

``` bash
git clone https://github.com/RachelZheng/scott.git
cd scott/
pip install -r requirement.txt
```

Our package also depends on [mop library(>=0.8)](https://bitbucket.org/suppechasper/optimaltransport/downloads/) on [R(>=3.0.0)](https://www.r-project.org/).

## Data Preparation

Put segmentation masks and the ground-truth mask of the first frame in the same folder. Convey folder name to pipeline.py using -i option.

### Naming Criteria:

+ Cell segmentation masks: uint16, name as **seg[INDEX, %03d].tif**. 
+ First ground-truth labeling mask: uint16, name as **man_track[INDEX, %03d].tif**. 

### Specifying Parameters:

Specify your parameters in "parameter_datasets" of [config.py](utils/config.py), where the parameters are:

+ n_pixels_per_cell: int, minimum number of pixels in each cell
+ mitosis_detection: bool, does cell have mitosis or not
+ expResizeWidth: initial image width
+ expResizeHeight: initial image height

If cells have mitosis behavior, please specify the mitosis parameters in line 18-22, where the default setup is for PSC-C2DL. We recommend not specifying weight in the first run and trying different mitosis parameters if accuracy improves.

## Usage

Run the pipeline.py file:

``` bash
cd scott/
python pipeline.py -i [path_seg] -o [path_out] -n [name_dataset]
```

For example:

``` bash
python pipeline.py -i 'data/PhC-C2DL-PSC/01_SEG/' -o 'data_out/' -n 'PhC-C2DL-PSC'
```

To resize the images into a smaller size in the pipeline, use -r option:

``` bash
python pipeline.py -i [path_seg] -o [path_out] -n [name_dataset] -r [on/off] -rw [resized_width] -rh [resized_heights]
```

To set the weight in WGWD of cell division detection, use -w option:
``` bash
python pipeline.py -i [path_seg] -o [path_out] -n [name_dataset] -w [weight]
```

## Acknowledgements

The code is built on many excellent papers and packages, which include:

+ [POT library](https://pot.readthedocs.io/en/stable/)
+ [Multi-scale OT solver](https://bitbucket.org/suppechasper/optimaltransport/src/master/)
+ [Gromov-Wasserstein averaging](https://github.com/gpeyre/2016-ICML-gromov-wasserstein)