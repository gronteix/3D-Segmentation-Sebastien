#!/usr/bin/env python

# -*- coding: utf-8 -*-



# ==================================================================

# SpheroidSegment

#  A spheroid segmentation tool custom-made for image analysis
# and graph construction.

#

#   Copyright 2019 Gustave Ronteix

#   MIT License

#

#   Source Repository: https://github.com/gronteix/3D-segmentation

# ==================================================================

import multiprocessing as mp
import os
import glob
import tqdm

import process

###### PARAMETERS ######

livePosition = 2
green = 1
orange = 0

channels = [livePosition, green, orange]

zRatio = 1/4
rNoyau = 6
dCells = 60

path = r'X:\Gustave\Experiments\Nuclei Segmentation\04072019\Seb\tif'

###### START OF STUDY ######

# process._sortFiles(path)

if __name__ == '__main__':

    output = mp.Queue()

    processes = [mp.Process(target=process._makeSphParrallel, args=(path, key, zRatio,
        rNoyau, dCells, channels)) for key in glob.glob(path + r'\\**\\**')]

    for p in processes:
        p.start()

    for p in tqdm(processes):
        p.join()
