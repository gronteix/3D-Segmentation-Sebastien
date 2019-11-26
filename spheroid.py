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

    def __init__(self, path, position, time, zRatio, rNoyau, dCells, pxtoum, minmass):

        self.Path = path
        self.Position = position
        self.Time = time
        self.ZRatio = zRatio
        self.RNoyau = rNoyau
        self.DCells = dCells
        self.Pxtoum = pxtoum
        self.CellNumber = 300
        self.NucImage = []
        self.NucFrame = pandas.DataFrame()
        self.BorderCrop = 0 # pixels cropped on border
        self.MinMass = 600000 # to check for different images
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

        df = trackpy.locate(self.NucImage[:,:,:], self.RNoyau, minmass=self.MinMass, maxsize=None, separation=self.RNoyau, noise_size=1,
                    smoothing_size=None, threshold=None, invert=False, percentile=64, topn=self.CellNumber,
                    preprocess=True, max_iterations=10, filter_before=None, filter_after=None, characterize=True,
                    engine='numba')

        df = df.loc[df['mass'] > self.MinMass]

        df =df.loc[((df['x'] - df['x'].mean())**2 < 4*df['x'].std()**2) &
          ((df['y'] - df['y'].mean())**2 < 4*df['y'].std()**2)]

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

        df = pandas.DataFrame()
        i = 0

        GreenConv = gaussian_filter(self.GreenImage, sigma=self.RNoyau)
        OrangeConv = gaussian_filter(self.OrangeImage, sigma = self.RNoyau)

        for cellLabel in self.Spheroid['cells'].keys():

            try:

                x = int(float(self.Spheroid['cells'][cellLabel]['x']))
                y = int(float(self.Spheroid['cells'][cellLabel]['y']))
                z = int(float(self.Spheroid['cells'][cellLabel]['z']))

                df.loc[i, 'label'] = cellLabel
                df.loc[i, 'Orange'] = OrangeConv[z,x,y]
                df.loc[i, 'Green'] = GreenConv[z,x,y]
                i += 1

            except Exception as e: print('Error in cell ' + str(cellLabel), e)

        # We experimentally classify the cells spheroid by spheroid

        X = df[['Orange', 'Green']]


        # Pre-processing of the fluorescence data values
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

        a =X[:,0].dot(X[:,1])/X[:,0].dot(X[:,0])
        df['Color'] = np.sign(X[:,1]-a*X[1,0])

        gmm =  mixture.GaussianMixture(n_components=2).fit(X)
        labels = gmm.predict(X)
        df['GMM Color'] = labels*2-1

        for cellLabel in self.Spheroid['cells'].keys():

            self.Spheroid['cells'][cellLabel]['Intensity Orange'] = df.loc[df['label'] == cellLabel, 'Orange'].iloc[0]

            self.Spheroid['cells'][cellLabel]['Intensity Green'] = df.loc[df['label'] == cellLabel, 'Green'].iloc[0]

            # Error can come from thrown out cells from above that are non existent
            # here...

            if df.loc[df['label'] == cellLabel, 'GMM Color'].iloc[0] < 0:

                self.Spheroid['cells'][cellLabel]['state GMM'] = 'Orange'

            if df.loc[df['label'] == cellLabel, 'GMM Color'].iloc[0] > 0:

                self.Spheroid['cells'][cellLabel]['state GMM'] = 'Green'

            if df.loc[df['label'] == cellLabel, 'Color'].iloc[0] < 0:

                self.Spheroid['cells'][cellLabel]['state linear'] = 'Orange'

            if df.loc[df['label'] == cellLabel, 'Color'].iloc[0] > 0:

                self.Spheroid['cells'][cellLabel]['state linear'] = 'Green'

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
            dic['neighbours'] = self._nearestNeighbour(df, label,
                                        dCells, zRatio)
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
        a,b,c = dCells

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

        img_eq = exposure.equalize_hist(self.OrangeImage)
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

                    #rloc = np.sqrt((r/self.Pxtoum)**2 - (z -n)**2/self.ZRatio**2)
                    #s = np.linspace(0, 2*np.pi, 100)
                    #x = rloc*np.sin(s) + x
                    #y = rloc*np.cos(s) + y

                    if self.Spheroid['cells'][cellLabel]['state GMM'] == 'Green':

                        plt.plot(x, y, 'xg')

                    elif self.Spheroid['cells'][cellLabel]['state GMM'] == 'Orange':

                        plt.plot(x, y, 'xr')

                    else: plt.plot(x, y, 'xr')

            plt.savefig(self.Path + r'/filmstack/im_' + str(n) +'.png')
            plt.close()
