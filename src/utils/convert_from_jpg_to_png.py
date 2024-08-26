import os
from PIL import Image
import numpy as np
DATA_FOLDER = os.path('../data')


def recurive_rename(folder):
    for element in folder:
        if os.path.isfile(element):
            os.rename(element, os.path.splitext(element)[0] + '.png')
        elif os.path.isdir(element):
            recurive_rename(element)

if __name__ == '__main__':
    recurive_rename(DATA_FOLDER)
            