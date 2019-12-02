from __future__ import (absolute_import, division, print_function,
    unicode_literals)

import os
import json
from tqdm import tqdm_notebook as tqdm
import numpy as np

from spheroid import spheroid

def default(o):
    if isinstance(o, np.int64): return int(o)
    raise TypeError

def _sortFiles(path):

    print('Please verify that the filename order is $t$ then $xy$ then $z$')

    for file in tqdm(os.listdir(path)):

        if not os.path.isdir(path + r'\\' + file):

            fileName, ending = file.split('.')

            if not 't' in fileName:

                _, position = fileName.split('xy')
                position, _ = position.split('z')

                time = '00'

            elif 't' in fileName:


                _, position = fileName.split('xy')
                position, time = position.split('t')
                time, z = time.split('z')


            if not os.path.exists(path + r'\\' + position):
                os.mkdir(path + r'\\' + position)

            if not os.path.exists(path + r'\\' + position + r'\\' + time):
                os.mkdir(path + r'\\' + position + r'\\' + time)

            os.rename(path + r'\\' + file, path + r'\\' + position + r'\\'
                + time + r'\\' + file)

def _saveSpheroid(sph, path):

    print(path)

    with open(path, 'w') as fp:

        json.dump(sph, fp, default = default)


def _makeSingleSpheroidClass(path, spheroidFolder, timeFolder, zRatio, rNoyau,
    dCells, pxtoum, channels, minmass):

    print('prep image: ' + spheroidFolder + ' folder and time ' + timeFolder)

    filePath =  os.path.join(path,spheroidFolder,timeFolder,'cropped')

    if not os.path.exists(os.path.join(path,'Spheroids')):
            os.mkdir(os.path.join(path,'Spheroids'))

    Sph = spheroid(filePath, spheroidFolder, timeFolder, zRatio, rNoyau, dCells,
                    pxtoum, minmass)
    # Initialize spheroid

    if len(channels) == 3: # Improve dependancy on channel number...

        Sph._loadImage(channels[0], 'NucImage') # Load live cells
        Sph._loadImage(channels[1], 'GreenImage') # Load green cells
        Sph._loadImage(channels[2], 'OrangeImage') # Load orange cells

    else: return print("Wrong number of color channels")

    print('image made, starting nuclei ID')

    Sph._getNuclei() # identification of nuclei positions

    print('nuclei gotten, make spheroid')

    Sph._makeSpheroid() # creation of dict object

    Sph.NucFrame.to_csv(path + r'\\' + 'Spheroids' +
        '\\SpheroidFrame_' + spheroidFolder + r'_' +  timeFolder + '.csv')

    print('refine the analysis over the spheroid')

    Sph._refineSph() # creation of dict object

    print('refined the spheroid properties')

    if len(channels) == 3:

        if not os.path.exists(path + r'\Spheroids'):
                os.mkdir(path + r'\\' + 'Spheroids')

        try:
            df = Sph._initializeStates()
            df.to_csv(path + r'\\' + 'Spheroids' +
                '\\intensityFrame_' + spheroidFolder + r'_' +  timeFolder + '.csv')
        except Exception as e: print(e)

    if not os.path.exists(path + r'\\' + 'Spheroids'):
            os.mkdir(path + r'\\' + 'Spheroids')

    print(path + r'\\' + 'Spheroids' +
            '\\spheroid_' + spheroidFolder + r'_' +  timeFolder + '.json')

    _saveSpheroid(Sph.Spheroid, path + r'\\' + 'Spheroids' +
            '\\spheroid_' + spheroidFolder + r'_' +  timeFolder + '.json')

    print('verif. sph.')

    #Sph._verifySegmentation()

def _makeSpheroidClass(path, zRatio, rNoyau, dCells, pxtoum, channels, minmass):

    """
    ====== COMMENT =======

    Function to be optimized for parrallization.
    """

    for spheroidFolder in tqdm(os.listdir(path)):

        spheroidPath = path + r'\\' + spheroidFolder

        if os.path.isdir(spheroidPath):

            for timeFolder in os.listdir(spheroidPath):

                timePath = spheroidPath + r'\\' + timeFolder

                if os.path.isdir(timePath):

                    try:

                        _makeSingleSpheroidClass(path, spheroidFolder,
                            timeFolder, zRatio, rNoyau, dCells, pxtoum, channels, minmass)

                    except Exception as e: print(e)

    return print('Spheroids made')

def _makeSphParrallel(path, key, zRatio, rNoyau, dCells, pxtoum,channels):

    """
    ====== COMMENT =======

    Function to be optimized for parrallization.
    """

    _, spheroidFolder = key.split(path)
    _, spheroidFolder, timeFolder = spheroidFolder.split('\\')

    _makeSingleSpheroidClass(path, spheroidFolder, timeFolder, zRatio, rNoyau,
        dCells, channels, pxtoum)

    return print('Spheroids made')
