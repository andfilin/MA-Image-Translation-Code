"""
functions for loading digitimages to be used for imagegenerators.
"""

from os import listdir
from pathlib import Path
import cv2

char74k_path = Path(__file__).parent.absolute() / "Chars74K/English/Fnt"

####
# load images of digits from char74k
####
# datasetPath: path to char74k
# font:  index of font to use
#
# returns: list of digitimages of that font
####
def load_char74k_digits(datasetPath=char74k_path, imread_mode=cv2.IMREAD_GRAYSCALE, font=28):
    # for every digit 0-9, collect folder containing images of that digit
    digitFolders = [
        datasetPath / ("Sample00" + str(digit)) for digit in range(1,10)
    ]
    digitFolders.append(datasetPath / "Sample010")
    
    # for every digit 0-9, collect every image of that digit in each font
    digitImagePaths = []
    for digitFolder in digitFolders:
        imagePaths = [digitFolder / imageName for imageName in listdir(digitFolder)]
        digitImagePaths.append(imagePaths) 
    
    # from every folder of digitimages, select that image with specified font
    digitImages = [
        cv2.imread(str(digitImagePaths[digit][font]), imread_mode)
        for digit in range(10)
    ]
    return digitImages
   