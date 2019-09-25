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

import trackpy

class spheroid:

    """ Spheroid class containing the necessary info to build the spheroid.

    ====== COMMENT ======

     - All variables starting with a capital letter refer to class variables.
     - All variables starting with '_' refer to a dict


    ====== PARAMETERS ======

    path: string object, path to the folder storing images
    position: string object, well ID
    time: string object, time of experiment"""

    def __init__(self, path, position, time, zRatio, rNoyau, dCells):

        position = '0'
        time = '0'

        self.Path = path
        self.Position = position
        self.Time = time
        self.ZRatio = zRatio
        self.RNoyau = rNoyau
        self.DCells = dCells
        self.NucImage = []
        self.BorderCrop = 0 # pixels cropped on border
        self.MinMass = 80000 # to check for different images
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

        self.NucFrame = trackpy.locate(self.NucImage[:, :,:], self.RNoyau, minmass=self.MinMass, maxsize=None, separation=self.DCells,
                            noise_size=2, smoothing_size=None, threshold=None, invert=False, percentile=64, topn=None,
                            preprocess=True, max_iterations=10, filter_before=None, filter_after=None, characterize=True,
                            engine='numba')

        self.NucFrame['label'] = range(len(fd))


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

        X = np.arange(0, 100)
        Y = np.arange(0, 100)
        Z = np.arange(0, 100)
        X, Y, Z = np.meshgrid(X, Y, Z)

        df = pandas.DataFrame()
        i = 0

        (z, x, y) = self.RNoyau

        mask = np.sqrt((X-50)**2/x**2 + (Y-50)**2/y**2 + (Z-50)**2/z**2) < 1
        mask = np.transpose(mask, (2,1,0)).astype(np.int)

        xmin = int(self.NucFrame['x'].min())
        xmax = int(self.NucFrame['x'].max())
        ymin = int(self.NucFrame['y'].min())
        ymax = int(self.NucFrame['y'].max())
        zmin = int(self.NucFrame['z'].min())
        zmax = int(self.NucFrame['z'].max())

        (Zshape, Xshape, Yshape) = np.shape(self.GreenImage)

        GreenConv = scipy.signal.fftconvolve(self.GreenImage[max(0,zmin-10):min(zmax+10, Zshape), max(0,xmin-50):min(xmax+50, Xshape),
                                                             max(0,ymin-50):min(ymax+50, Yshape)], mask, mode='full')
        OrangeConv = scipy.signal.fftconvolve(self.OrangeImage[max(0,zmin-10):min(zmax+10, Zshape), max(0,xmin-50):min(xmax+50, Xshape),
                                                             max(0,ymin-50):min(ymax+50, Yshape)], mask, mode='full')

        for cellLabel in tqdm(self.Spheroid['cells'].keys()):

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
        dCells = self.DCells
        zRatio = self.ZRatio

        df['label'] = df['label'].astype(int).astype(str)

        for label in df['label'].unique():

            dic = {}

            # All values are strings since json doesn't know ints

            dic['x'] = str(df.loc[df['label'] == label, 'x'].iloc[0])
            dic['y'] = str(df.loc[df['label'] == label, 'y'].iloc[0])
            dic['z'] = str(df.loc[df['label'] == label, 'z'].iloc[0])
            dic['neighbours'] = self._nearestNeighbour(df, label, dCells, zRatio)
            dic['state'] = 'Live'

            _Cells[str(int(label))] = dic

        return _Cells


    def _nearestNeighbour(self, df, label, dCells, zRatio):

        """Returns a list of float labels of the cells closest to the given label.
        This method is dependant only on a minimum distance given by the investigator."""

        x = df.loc[df['label'] == label, 'x'].iloc[0]
        y = df.loc[df['label'] == label, 'y'].iloc[0]
        z = df.loc[df['label'] == label, 'z'].iloc[0]

        lf = df.loc[df['label'] != label].copy()

        return lf.loc[np.sqrt((lf['x'] - x)**2 + (lf['y'] - y)**2 +
            (lf['z'] - z)**2/zRatio**2) < dCells, 'label'].values.tolist()


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

        self.Spheroid['N'] = len(self.Spheroid['cells'])
        self.Spheroid['assortativity'] = nx.degree_assortativity_coefficient(G)
        self.Spheroid['average degree'] = np.asarray([float(C[v]) for v in G]).mean()


    def _verifySegmentation(self):

        if not len(self.NucFrame):
            return print('Image doesnt exist')

        if not os.path.exists(self.Path + r'/filmstack/'):
            os.mkdir(self.Path + r'/filmstack/')

        zshape, _, _ = np.shape(self.NucImage)

        ImageAll = self.NucImage

        for n in range(zshape):

            Image = ImageAll[n]

            plt.figure(figsize=(12, 12))
            plt.subplot(111)
            plt.imshow(Image, vmin=0, vmax=1300, cmap=plt.cm.gray)
            plt.axis('off')

            #scalebar = ScaleBar(0.0000003, location = 'lower right')
            # 1 pixel = 0.3 um
            #plt.gca().add_artist(scalebar)

            r = self.RNoyau

            for cellLabel in self.Spheroid['cells'].keys():

                x = int(self.Spheroid['cells'][cellLabel]['x'])
                y = int(self.Spheroid['cells'][cellLabel]['y'])
                z = int(self.Spheroid['cells'][cellLabel]['z'])

                if (r**2 - (z -n)**2/self.ZRatio**2) > 0:

                    rloc = np.sqrt(r**2 - (z -n)**2/self.ZRatio**2)
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
