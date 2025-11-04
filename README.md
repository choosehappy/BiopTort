# BiopTort

BiopTort is a tool designed for quantifying tortuosity in core needle biopsies. Tortuosity is a measure of the twisting and turning of the biopsy path, and is a common yet largely unreported artifact in pathology.

## Features
BiopTort produces two types of output:

1. A `.pptx` file where each slide displays the qualitative BiopTort output:
![alt text](bioptort/figures/example_pptx.png)

2. A `.csv` file containing quantitative metrics extracted for each slide:
![alt text](bioptort/figures/example_csv.png)

## Installation
### Using Docker
```bash
docker run -v <YOUR_DATA_PATH>:/data -it jacksonjacobs1/bioptort:main /bin/bash
```

## Usage
BiopTort exposes the following CLI options:
```bash
$ python3 -m bioptort --help
usage: __main__.py [-h] --im_path IM_PATH --ppt_save PPT_SAVE [--csv_save CSV_SAVE] [--mask_path MASK_PATH] [--sort] [--gauss_sigma GAUSS_SIGMA]

options:
  -h, --help            show this help message and exit
  --im_path IM_PATH     The directory containing (openslide-compatible) CNB WSIs.
  --ppt_save PPT_SAVE   The path to save the powerpoint presentation, including the filename.
  --csv_save CSV_SAVE   Optional path to save a csv file containing quantitative results, including the filename.
  --mask_path MASK_PATH
                        Optional path to existing png tissue masks for the CNB WSIs with matching filenames.
  --sort, -s            Sort the slides in order of highest to lowest tortuosity.
  --gauss_sigma GAUSS_SIGMA
                        The sigma value for the gaussian filter.
```


```bash
python3 -m bioptort --im_path /data/ --ppt_path /data/<YOUR_PPT_NAME>.pptx --csv_path /data/<YOUR_CSV_NAME>.csv
```

## Citation
The related paper is published in the Archives of Pathology and Laboratory Medicine:

[https://meridian.allenpress.com/aplm/article/doi/10.5858/arpa.2025-0028-OA/508037](https://meridian.allenpress.com/aplm/article/doi/10.5858/arpa.2025-0028-OA/508037)

Please use below to cite this paper if you find this repository useful or if you use the software shared here in your research.

```
Jacobs, J., Wiener, D., Pathak, T., Farré, X., Mirtti, T., & Janowczyk, A. (2025). Standardized Reporting of Core Needle Biopsy TortuosityBiopTort—A Computer-Aided Protocol for Categorizing Tortuosity. Archives of Pathology & Laboratory Medicine. https://doi.org/10.5858/ARPA.2025-0028-OA
```

```
@article{Jacobs2025,
   author = {Jackson Jacobs and Dan Wiener and Tilak Pathak and Xavier Farré and Tuomas Mirtti and Andrew Janowczyk},
   doi = {10.5858/ARPA.2025-0028-OA},
   issn = {1543-2165},
   journal = {Archives of Pathology \& Laboratory Medicine},
   month = {10},
   title = {Standardized Reporting of Core Needle Biopsy TortuosityBiopTort—A Computer-Aided Protocol for Categorizing Tortuosity},
   url = {https://dx.doi.org/10.5858/arpa.2025-0028-OA},
   year = {2025}
}

```
