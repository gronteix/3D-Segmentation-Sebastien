from __future__ import (absolute_import, division, print_function,
    unicode_literals)

import os
import numpy as np
import pandas
import os
import skimage
from scipy import ndimage, signal
from skimage import exposure
import matplotlib.pyplot as plt
from skimage import io
import scipy
from scipy import signal
from skimage import feature
import json
from matplotlib_scalebar.scalebar import ScaleBar
import networkx as nx
from sklearn import mixture
from scipy.ndimage import gaussian_filter

import trackpy

class spheroid:

    """ Spheroid class containing the necessary info to build the spheroid.

    ====== COMMENT ======

     - All variables starting with a capital letter refer to class variables.
     - All variables starting with '_' refer to a dict
     - All distances and sizes are 'real-world', the pxtoum serves to convert
        them to px


    ====== PARAMETERS ======

    path: string object, path to the folder storing images
    position: string object, well ID
    time: string object, time of experiment"""

    def __init__(self, path, position, time, zRatio, rNoyau, dCells, pxtoum):

        self.Path = path
        self.Position = position
        self.Time = time
        self.ZRatio = zRatio
        self.RNoyau = rNoyau
        self.DCells = dCells
        self.Pxtoum = pxtoum
        self.CellNumber = 200
        self.NucImage = []
        self.NucFrame = pandas.DataFrame()
        self.BorderCrop = 0 # pixels cropped on border
        self.MinMass = 50000 # to check for different images
        self.ThreshOrange = 300 # thresh for orange cell detection, not used since
                                # classifier introduced.
        self.ThreshGreen = 200  # thresh for orange cell detection, not used
                                # since classifier introduced
        self.ThreshCell = 550   # thresh for live cell detection

    def _loadImage(self, channel, type):

        """ Function to load the images corresponding to a given channel

        ====== COMMENT ======

        The function needs to be improved so as to add new channels without
        requiring to manually add new channels by hand.

        """

        image_list = []
        for filename in sorted(os.listdir(self.Path)): #assuming tif

            if '.tif' in filename:

                im = io.imread(self.Path + '/' + filename)

                image_list.append(im[:,:,channel])

        if type == 'NucImage':

            self.NucImage = np.asarray(image_list)

        if type == 'GreenImage':

            self.GreenImage = np.asarray(image_list)

        if type == 'OrangeImage':

            self.OrangeImage = np.asarray(image_list)

    def _getNuclei(self):

        """
        Creates the dataframe containing all the cells of the Spheroid.
        The duplicata clean function is eliminated. Indeed, the local maximum
        makes it impossible for any cell to be segmented twice along the z-axis.
        """

        dz, dx, dy = self.RNoyau

        dX = 2*(int(dx/self.Pxtoum)//2)+1
        dY = 2*(int(dy/self.Pxtoum)//2)+1
        dZ = 2*(int(dz/self.Pxtoum)//2)+1

        r = (dZ, dX, dY)

        print(self.RNoyau)
        print(r)

        dz, dx, dy = self.DCells

        dX = 2*(int(dx/self.Pxtoum)//2)+1
        dY = 2*(int(dy/self.Pxtoum)//2)+1
        dZ = 2*(int(dz/self.Pxtoum)//2)+1

        d = (dZ, dX, dY)

        print(self.DCells)
        print(d)

        df = trackpy.locate(self.NucImage[:,:,:], r, minmass=None, maxsize=None, separation=None, noise_size=1,
                    smoothing_size=None, threshold=None, invert=False, percentile=64, topn=self.CellNumber,
                    preprocess=True, max_iterations=10, filter_before=None, filter_after=None, characterize=True,
                    engine='numba')

        df = df.loc[df['mass'] > self.MinMass]
        df['label'] = range(len(df))

        self.NucFrame = df



    def _makeSpheroid(self):

        """Generates the spheroid dict containing all the essential information about the spheroid.

        ====== PARAMETERS ======

        df: DataFrame containing all the positional information of the cells
        dCells: maximum distance for two cells to be considered as neighbours
        zRatio: pixel ratio between z and xy dimensions
        Image: original, multichannel, 3D image
        state: True/False variable stating if we seek the dead-alive information

        ====== RETURNS ======

        _Spheroid: dict object"""

        _Spheroid = {}

        _Spheroid['spheroid position'] = self.Position
        _Spheroid['time'] = self.Time
        _Spheroid['cells'] = self._generateCells()

        self.Spheroid = _Spheroid


    def _initializeStates(self):

        import cv2

        dz, dx, dy = self.RNoyau

        X = np.arange(0, int(dx*6/self.Pxtoum))
        Y = np.arange(0, int(dy*6/self.Pxtoum))
        Z = np.arange(0, int(dz*6/self.Pxtoum))
        X, Y, Z = np.meshgrid(X, Y, Z)

        df = pandas.DataFrame()
        i = 0

        (z, x, y) = self.RNoyau
        z /= self.Pxtoum
        x /= self.Pxtoum
        y /= self.Pxtoum

        mask = np.sqrt((X-dx/2)**2/x**2 + (Y-dy/2)**2/y**2 + (Z-dz/2)**2/z**2) < 1
        mask = np.transpose(mask, (2,1,0)).astype(np.int)

        import cv2

        GreenConv = scipy.signal.fftconvolve(self.GreenImage, mask, mode='full')
        OrangeConv = scipy.signal.fftconvolve(self.OrangeImage, mask, mode='full')

        for cellLabel in self.Spheroid['cells'].keys():

            try:

                x = int(float(self.Spheroid['cells'][cellLabel]['x']))
                y = int(float(self.Spheroid['cells'][cellLabel]['y']))
                z = int(float(self.Spheroid['cells'][cellLabel]['z']))

                zlen, _, _ = np.nonzero(mask)

                df.loc[i, 'label'] = cellLabel
                df.loc[i, 'Orange'] = OrangeConv[z,x,y]/len(zlen)
                df.loc[i, 'Green'] = GreenConv[z,x,y]/len(zlen)
                i += 1

            except Exception as e: print('Error in cell ' + str(cellLabel), e)

        # We experimentally classify the cells spheroid by spheroid

        a = df['Orange'].dot(df['Green'])/df['Orange'].dot(df['Orange'])
        df['Color'] = np.sign(df['Green']-a*df['Orange'])

        X = df[['Orange', 'Green']]
        gmm =  mixture.GaussianMixture(n_components=2).fit(X)
        labels = gmm.predict(X)
        df['GMM Color'] = labels*2-1

        for cellLabel in self.Spheroid['cells'].keys():

            # Error can come from thrown out cells from above that are non existent
            # here...

            try:

                if df.loc[df['label'] == cellLabel, 'GMM Color'].iloc[0] < 0:

                    self.Spheroid['cells'][cellLabel]['state'] = 'Orange'

                if df.loc[df['label'] == cellLabel, 'GMM Color'].iloc[0] > 0:

                    self.Spheroid['cells'][cellLabel]['state'] = 'Green'

            except Exception as e: print('Error in cell ' + str(cellLabel), e)

        return df


    def _generateCells(self):

        """ This function serves to generate the cells and gives back a dic object.

        ====== PARAMETERS ======

        df: DataFrame containing the relevant positional information
        dCells: minimum distance between any two cells
        zRatio: ratio between the xy and z dimensions

        ====== RETURNS ======

        _Cells: dict object"""

        _Cells = {}

        df = self.NucFrame
        rCells = self.RNoyau
        zRatio = self.ZRatio

        df['label'] = df['label'].astype(int).astype(str)

        for label in df['label'].unique():

            dic = {}

            # All values are strings since json doesn't know ints

            dic['x'] = str(df.loc[df['label'] == label, 'x'].iloc[0])
            dic['y'] = str(df.loc[df['label'] == label, 'y'].iloc[0])
            dic['z'] = str(df.loc[df['label'] == label, 'z'].iloc[0])
            dic['neighbours'] = self._nearestNeighbour(df, label,
                                        rCells, zRatio)
            dic['state'] = 'Live'

            _Cells[str(int(label))] = dic

        return _Cells


    def _nearestNeighbour(self, df, label, dCells, zRatio):

        """Returns a list of float labels of the cells closest to the given label.
        This method is dependant only on a minimum distance given by the investigator."""

        x = df.loc[df['label'] == label, 'x'].iloc[0]
        y = df.loc[df['label'] == label, 'y'].iloc[0]
        z = df.loc[df['label'] == label, 'z'].iloc[0]

        # We choose the neighbours less than 2 cell distances away
        (a,b,c) = dCells
        a *= 2/self.Pxtoum
        b *= 2/self.Pxtoum
        c *= 2/self.Pxtoum

        lf = df.loc[df['label'] != label].copy()

        return lf.loc[np.sqrt((lf['x'] - x)**2/b**2 + (lf['y'] - y)**2/c**2 +
            (lf['z'] - z)**2/a**2) < 1, 'label'].values.tolist()


    def _makeG(self):

        G=nx.Graph()
        _Cells = self.Spheroid['cells']
        G.add_nodes_from(_Cells.keys())

        for key in _Cells.keys():

            neighbours = _Cells[key]['neighbours']

            for node in neighbours:

                G.add_edge(key, node)

        return G


    def _refineSph(self):

        G = self._makeG()

        A = nx.betweenness_centrality(G) # betweeness centrality
        B = nx.clustering(G)
        C = nx.degree(G)

        for v in G:

            self.Spheroid['cells'][v]['degree'] = C[v]
            self.Spheroid['cells'][v]['clustering'] = B[v]
            self.Spheroid['cells'][v]['centrality'] = A[v]

    def _verifySegmentation(self):

        from skimage import exposure

        if not len(self.NucFrame):
            return print('Image doesnt exist')

        if not os.path.exists(self.Path + r'/filmstack/'):
            os.mkdir(self.Path + r'/filmstack/')

        zshape, _, _ = np.shape(self.NucImage)

        img_eq = exposure.equalize_hist(self.NucImage)
        ImageAll =  gaussian_filter(img_eq, sigma=2)

        for n in range(zshape):

            Image = ImageAll[n]

            plt.figure(figsize=(12, 12))
            plt.subplot(111)
            plt.imshow(Image, cmap=plt.cm.gray)
            plt.axis('off')

            #scalebar = ScaleBar(0.0000003, location = 'lower right')
            # 1 pixel = 0.3 um
            #plt.gca().add_artist(scalebar)

            (_,r,_) = self.RNoyau

            for cellLabel in self.Spheroid['cells'].keys():

                x = int(float(self.Spheroid['cells'][cellLabel]['x']))
                y = int(float(self.Spheroid['cells'][cellLabel]['y']))
                z = int(float(self.Spheroid['cells'][cellLabel]['z']))

                if ((r/self.Pxtoum)**2 - (z -n)**2/self.ZRatio**2) > 0:

                    rloc = np.sqrt((r/self.Pxtoum)**2 - (z -n)**2/self.ZRatio**2)
                    s = np.linspace(0, 2*np.pi, 100)
                    x = rloc*np.sin(s) + x
                    y = rloc*np.cos(s) + y

                    if self.Spheroid['cells'][cellLabel]['state'] == 'Green':

                        plt.plot(y, x, 'g-')

                    elif self.Spheroid['cells'][cellLabel]['state'] == 'Orange':

                        plt.plot(y, x, 'r-')

                    else: plt.plot(self.Spheroid['cells'][cellLabel]['y'],
                        self.Spheroid['cells'][cellLabel]['x'], 'r.')

            plt.savefig(self.Path + r'/filmstack/im_' + str(n) +'.png')
            plt.close()
