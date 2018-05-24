#!/usr/bin/env python

import os
import sys
import time
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Series
import pims
import trackpy as tp
import warnings
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import axes3d, Axes3D
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg
pg.setConfigOption('useOpenGL', False)


def preprocess(idir):
    ''' Loads images then runs the preprocessing part of SEEVIS
        idir: path containing the 3 channels (RGB)
        exports the resulting image into the default seevis_output folder
    '''
    # load imgs
    f_red, f_green, f_blue = get_imlist(idir)
    print "\n%d files found\t" % (len(f_red)*3)
    print "Loading data..."
    red, size_var = load_img(f_red)
    # 1st frame properties (rows, cols)/(height, width)
    rows, cols, channels = size_var.shape
    print "Image size ", size_var.shape
    green, sv = load_img(f_green)
    blue, sv = load_img(f_blue)
    # enhancing the
    rgb, ug, uclahe, ctrast, dblur, mblur, tmask, res = approach(red, blue, green)
    # default export
    timestr = time.strftime("%Y%m%d-%H%M%S")
    outdir = timestr + "seevis_output"
    out = [res]
    dirs = [outdir]
    export(f_red, dirs, out)
    return outdir


def get_data(outdir):
    ''' Loads the output of the preprocessing steps for feature extraction
        Returns the formatted data
    '''
    frames = pims.ImageSequence("../"+outdir+"/*tif")
    print frames

    # particle diameter
    diam = 11
    features = tp.batch(frames[:frames._count], diameter=diam, minmass=1, invert=True)
    # Link features in time: sigma_(max)
    search_range = diam-2
    # r, g, b images are loaded
    lframes = int(np.floor(frames._count/3))
    # default max 15% frame count
    imax = int(np.floor(15*lframes/100))
    t = tp.link_df(features, search_range, memory=imax)
    # default neighbour strategy: KDTree

    # Filter spurious trajectories
    # default min 10% frame count
    imin = int(np.floor(10*lframes/100))
    # if seen in imin
    t1 = tp.filter_stubs(t, imin)

    # Compare the number of particles in the unfiltered and filtered data.
    print 'Unique number of particles (Before filtering):', t['particle'].nunique()
    print '(After):', t1['particle'].nunique()

    # export pandas data frame with filename being current date and time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    data = pd.DataFrame({'x': t1.x, 'y': t1.y, 'z': t1.frame, 'mass': t1.mass, 'size': t1.size, 'ecc': t1.ecc, 'signal': t1.signal, 'ep': t1.ep, 'particle': t1.particle})

    file_name = "../features_" + timestr + ".csv"
    print "Exporting %s" % (file_name)
    data.to_csv(file_name, sep='\t', encoding='utf-8')
    return data


def visualise(data, s):
    ''' Visualise one of the 4 schemes included in SEEVIS
        Args    the dataframe (see get_data) and s, the supplied scheme (int)
        displays directly the user-requested vis.
    '''
    # Prepare the data for a 3D scatter plot
    ld = len(data)
    # n of unique particles
    n = data['particle'].nunique()
    pos = reshape_xyz(data.x.values, data.y.values, data.z.values, ld)
    size = np.repeat(3, ld)
    # initialise colors based on the user-selected scheme
    if s == 1:
        c = nm(n, data)
        display(data, c, size, pos)
    elif s == 2:
        c = tm(ld, data, pos)
        display(data, c, size, pos)
    else:
        c, data = pm(data)
        display(data, c, size, pos)


def elapsed_time(start_time):
    print("\t--- %4s seconds ---\n" % (time.time()-start_time))


def create_dir(dir):
    ''' Directory creation
    '''
    if not os.path.exists(dir):
        os.makedirs(dir)
    print "Directory '%s' created" % (str(dir))


def get_imlist(path):
    ''' Returns a list of filenames for all compatible extensions in a directory
        Args : path of the directory and the supplied user choice
        Handles dir with i_ext as file extension
        c2, c3, c4 as red, green, blue channels respectively
        Returns the list of files to be treated
    '''
    i_ext = tuple([".tif", ".jpg", ".jpeg", ".png"])
    flist = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(i_ext)]
    if len(flist) is not None:
        ext = str.split(flist[0], ".")[-1]
        f_red = [r for r in flist if r.endswith("c2" + "." + ext)]
        f_green = [g for g in flist if g.endswith("c3" + "." + ext)]
        f_blue = [b for b in flist if b.endswith("c4" + "." + ext)]
        if len(f_red) is None or len(f_red) == 0:
            print "error: image filenames do not comply. Image filenames must be formatted as follows: red, c1.tif; )"
            sys.exit(1)
        return f_red, f_green, f_blue
    else:
        warnings.filterwarnings("ignore")
        print 'Directory contains unsupported files. Please refer to the README file)'
        sys.exit(1)


def load_img(flist):
    ''' Loads images in a list of arrays
        Args : list of files
        Returns list of all the ndimage arrays
    '''
    imgs = []
    for i in flist:
        # return img as is
        imgs.append(cv2.imread(i, -1))
    size_var = cv2.imread(i)
    return imgs, size_var


def rgb_to_gray(img):
    ''' Converts an RGB image to grayscale, where each pixel
        now represents the intensity of the original image.
    '''
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def invert(img):
    ''' Inverts an image
    '''
    cimg = np.copy(img)
    return cv2.bitwise_not(cimg, cimg)


def ubyte(img):
    return cv2.convertScaleAbs(img, alpha=(255.0/65535.0))


def reshape_xyz(x, y, z, ld):
    ''' Reshapes coordinates
        Into an array( [ [x_i, y_i, z_i ], ..] with i from [0-> n-1]
    '''
    pos = []
    for i in range(ld):
        t = list(np.append(np.append(x[i], y[i]), z[i]))
        pos.append(t)
    return np.array(pos)


def invert(img):
    cimg = np.copy(img)
    return cv2.bitwise_not(cimg, cimg)


def approach(red, blue, green):
    ''' The SEEVIS implementation of the preprocessing steps
        Args    red, blue, green channels
        Returns 8 different variables highlighting the most important steps
    '''
    rgb, fgray, ug, uclahe, ctrast, dblur, mblur, tmask, res, res2 = [], [], [], [], [], [], [], [], [], []
    print "Preprocessing images..."
    # Parameters for manipulating image data
    # maxIntensity depends on dtype of image data
    maxIntensity = 255.0
    x = np.arange(maxIntensity)
    phi, theta = 1, 1
    for i in range(len(red)):

        # Add up 3C into RGB
        tmp_rgb = red[i] + green[i] + blue[i]
        rgb.append(tmp_rgb)

        # 1. PREPROCESSING FOR SIGNAL ENHANCEMENT #
        ###########################################
        # RGB to Gray
        tmp_gray = rgb_to_gray(tmp_rgb)
        fgray.append(tmp_gray)
        # uint16 to inverted ubyte
        tmp_ug = invert(ubyte(tmp_gray))
        ug.append(tmp_ug)
        # CLAHE 3x3
        gclahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
        tmp_uclahe = gclahe.apply(tmp_ug)
        uclahe.append(tmp_uclahe)
        # contrast enhanced picture
        tmp_ctrast = (maxIntensity/phi)*(tmp_uclahe/(maxIntensity/theta))**2
        tmp_ctrast = np.array(tmp_ctrast, dtype="uint8")
        ctrast.append(tmp_ctrast)

        # 2. Subtract Single bacterial signal
        ###########################################
        # Signal enhancement
        sblur = cv2.bilateralFilter(tmp_ctrast, 5, 75, 75)
        dblur.append(sblur)
        # Adaptive thresholding
        tmp_ssignal = cv2.adaptiveThreshold(sblur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)
        tmask.append(tmp_ssignal)

        # 3. Adaptively mask the signal region
        ###########################################
        # Median blur (rblur for FG signal region)
        rblur = cv2.medianBlur(tmp_ctrast, 15)
        mblur.append(rblur)
        # Threshold signal area
        ret, thresh2 = cv2.threshold(rblur, 225, 255, cv2.THRESH_BINARY)
        # Foreground signal
        mask = np.ones(red[0].shape[:2], dtype="uint8") * 255
        tmp_res = cv2.bitwise_or(thresh2, tmp_ssignal, mask=mask)
        res.append(tmp_res)

    return rgb, ug, uclahe, ctrast, dblur, mblur, tmask, res


def export(flist, dirs, out):
    ''' Enumerates elements in dirs and formats the filename depending on flist
    '''
    for j in enumerate(dirs):
        # create output dir
        create_dir("../" + j[1])
        for i in range(len(flist)):
            f = "../" + j[1] + "/" + j[1].split(" - ")[-1] + "_" + flist[i].split("/")[-1]
            tmp = out[j[0]][i]
            cv2.imwrite(f, tmp)


def get_cmap(N, map):
    ''' Returns a function that maps each index in 0, 1, ... N-1 to a distinct
        RGB color. Uses a color palette and Mappable scalar
    '''
    color_norm = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap=map)
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color


def nm(n, data):
    ''' Nominal Mapping using the Tableau 10 palette
        Dissociate neighboring particles colors
        Returns a numpy array of N distinct colors cycled for n particles
        Args    n total unique particles and data, the dataframe
    '''
    # Amount of distinct colors to be used
    N = 10
    tableau10 = [(31, 119, 180), (255, 127, 14),
                 (44, 160, 44), (214, 39, 40),
                 (148, 103, 189), (140, 86, 75),
                 (227, 119, 194), (127, 127, 127),
                 (188, 189, 34), (23, 190, 207)]
    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
    for i in range(len(tableau10)):
        r, g, b = tableau10[i]
        tableau10[i] = (r / 255., g / 255., b / 255., 1.0)
    col = []
    for i in range(N):
        col.append(tableau10[i])
    # Cycle the same distinct 30 colors over the n dis particles
    L = []
    [L.extend(col) for i in xrange(n/N)]
    # add the rest to the list to obtain the same n of colors
    L.extend(col[0:n-len(L)])
    # Cycle through colors for all coord. pertaining to each n particles seen ld
    T = []
    # for each unique particle (particle id) count and store in c how many x times pid is seen
    for i in xrange(n):
        pid = np.unique(data.particle)[i]
        x = len(data[data.particle == pid])
        # color line i
        c = L[i]
        # extend T with the x repeated list
        T.extend([c]*x)
    # convert to numpy array for visualisation
    colors = []
    colors = np.array(T)
    return colors


def tm(n, data, pos):
    ''' color using time axis (data.z)
        cmap is viridis (spectral: black to white)
    '''
    N = data['z'].nunique()
    cmap = cm = get_cmap(N, 'viridis')
    col = []
    [col.append(cmap(i)) for i in range(N)]
    # cycle through N (pos[i][2]) time points/colors for all ld coord.
    L = []
    [L.append(col[pos[i][2].astype(int)]) for i in xrange(n)]
    colors = []
    colors = np.array(L)
    return colors


def find_missing(integers_list, start=None, limit=None):
    ''' Given a list of integers and optionally a start and an end
        finds all integers from start to end that are not in the list
    '''
    start = start if start is not None else integers_list[0]
    limit = limit if limit is not None else integers_list[-1]
    return [i for i in range(start, limit + 1) if i not in integers_list]


def pm(data):
    ''' Progeny Mapping by coloring progeny's single particles
        traced back to parent cells
    '''
    m = np.unique(data[data.z == np.max(data.z)].particle).astype(int)
    # select all pts pertaining to the list of m particles
    data1 = data[data['particle'].isin(list(m))]
    colors1 = nm(len(m), data1)
    # rest of particles shrinked to a smaller size and gray-colored
    # gray color
    c = np.array([1, 1, 1, .1])
    # get IDs of all missing particles
    o = find_missing(m)
    data2 = data[data['particle'].isin(list(o))]
    colors2 = np.tile(c, (len(data2), 1))
    colors = np.concatenate((colors1, colors2), axis=0)
    f = [data1, data2]
    datares = pd.concat(f)
    return colors, datares


def mkQApp():
    ''' Initialise an OpenGL 3D space / GUI for the visualisation
        with an optional distance to the view
    '''
    global QAPP
    QtGui.QApplication.setGraphicsSystem('raster')
    # work around a variety of bugs in the native graphics system
    inst = QtGui.QApplication.instance()
    if inst is None:
        QAPP = QtGui.QApplication([])
    else:
        QAPP = inst
    return QAPP


def load_data(path):
    data = pd.read_csv(path, index_col=0, parse_dates=True, sep='\t')
    data = pd.DataFrame({'x': data.x, 'y': data.y, 'z': data.z, 'mass': data.mass, 'size': data.size, 'data': data.ecc, 'signal': data.signal, 'ep': data.ep, 'particle': data.particle})
    return data


def display(data, color, size, pos):
    ''' Displays using pyqtgraph the 3D scatterplot with the preformatted color
        Args    data, dataframe
                pos, features' positions
                size of dots in the scatterplot
                color, the user-specific scheme
    '''
    app = pg.mkQApp()
    # Window widget
    w = gl.GLViewWidget()
    w.opts['distance'] = 1000
    w.resize(800, 800)
    w.show()
    w.setWindowTitle('SEEVIS - Features 3D scatterplot')
    # Base grid for the 3D space
    g = gl.GLGridItem()
    g.scale(50, 50, 50)
    w.addItem(g)
    #
    sp = gl.GLScatterPlotItem(pos=pos, size=size, color=color, pxMode=False)
    # center the vis to the first seen feature
    sp.translate(-pos[0][0], -pos[0][1], -pos[0][2])
    w.setCameraPosition(azimuth=-180, elevation=90)
    w.addItem(sp)
