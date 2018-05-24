# SEEVIS, (S)egmentation-Fr(EE) (VIS)ualization

A biomovie is a temporal series of digital microscopy images that is recorded for one selected visual field in a bacterial time course experiment. We present SEEVIS, a data driven (S)egmentation-fr(EE) and automatic pipeline of methods to (VIS)ualise the growth patterns of a cell population conveyed in a biomovie. It consists of three steps 1. signal enhancement to help adjust the signal-to-noise ratio, 2. feature detection so as to draw out qualitative information about the colony development, and 3. the visualization of the colony in a feature space as a 3D space-time cube supported with three appropriate colour coding methods. These codings are based on two colour palettes (```Tableau10``` and ```Viridis```), are titled and functionally described as follows:

1. Nominal colour coding (NCC) highlights single feature trajectories (```Tableau10```)
2. Time colour coding (TCC) visually promotes the extent of the population growth over time (```Viridis```)
3. Progeny colour coding (PCC) supports the process of tracing back features to their parents (```Tableau10```).

![Result](https://2.bp.blogspot.com/-OqaQtKtbZPo/VykvVJEa4YI/AAAAAAAAHv0/JkJ7kpnkfGAshGRJbA0OynaLXkLIURcpwCLcB/s1600/2.png "SEEVIS result for D1")
>*SEEVIS colour codings demonstrated for dataset D1. Figure (a) showcases the last frame of the biomovie, as a binary image after signal enhancement. The 3D visualization is displayed as 2D, azimuth = 0°, and elevation = 90° for the three colour codings: NCC (b), TCC (c), and PCC (d), respectively.*

This work was funded by the German-Canadian DFG International Research Training Group GRK 1906/1 and the “Phenotypic Heterogeneity and Sociobiology of Bacterial Populations” DFG SPP1617.

## Data

The employed datasets are available under The Open Data Commons Attribution License (ODC-By) v1.0.

Schlueter, J. - P., McIntosh, M., Hattab, G., Nattkemper, T. W., and Becker, A. (2015). Phase Contrast and Fluorescence Bacterial Time-Lapse Microscopy Image Data. Bielefeld University. [doi:10.4119/unibi/2777409](http://doi.org/10.4119/unibi/2777409).

## Dependencies

For better reproducibility the versions that were used for development are mentioned in parentheses.

* Python (2.7.11)
* matplotlib (2.2.2)
* OpenCV (2.4.12)
* pyqtgraph (0.10)
* trackpy (u'0.3.0rc1')
* pims (0.2.2)
* pandas (0.16.2)

## Usage

```bash
# Set file permissions
$ chmod +x seevis.py 

# Run SEEVIS on a folder containing all image files 
# Formatted by channel : red, green, blue as c2, c3, c4 respectively for every time point
$ ./seevis.py -i img_directory/

# Or on a CSV file containing feature positions
$ ./seevis.py -f filename.csv -s 2

#  -h, --help            show this help message and exit
#  -v, --version         show program's version number and exit
#  -i, --input           run SEEVIS on the supplied directory
#  -f, --file            run the Visualization of SEEVIS
#  -s                    run scheme (or colour coding) ranging from 1 to 4 (default is 1)
```

## License
```
The MIT License (MIT)

Copyright (c) Georges Hattab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. 
```
