"""
functions for loading datasets to be used for modeltraining.
"""

from os import listdir
from cv2 import imread, IMREAD_GRAYSCALE, IMREAD_COLOR, resize, BORDER_CONSTANT, copyMakeBorder
from numpy.random import shuffle
import pandas as pd
from pathlib import Path
import numpy as np

WMN_PATH = Path(__file__).parent.absolute() / "SCUT-WMN DataSet"

####
# resize image.
# either return cv2.resize(image, dims) if ratio is not to be kept,
# else scale one dimension and pad other.
####
def resize_image(image, dims, keepRatio=False):
    if not keepRatio:
        return resize(image, dims)
    
    targetWidth = dims[0]
    targetHeight = dims[1]
    
    inputWidth = image.shape[1]
    inputHeight = image.shape[0]
    # scale either width or height, depending on which scaling factor would be smaller
    scale_width = targetWidth / inputWidth
    scale_height = targetHeight / inputHeight
    
    if scale_width < scale_height:
        # scale width, pad height
        result = resize(image, dsize=(0,0), fx=scale_width, fy=scale_width)
        padding = targetHeight - result.shape[0]
        p_top = int(padding/2)
        p_bot = p_top if (padding%2) == 0 else p_top + 1
        assert padding >= 0 and (p_top + p_bot) == padding, "unexpected height-padding: %d"%(padding)
        result = copyMakeBorder(result, top=p_top, bottom=p_bot, left=0, right=0, borderType=BORDER_CONSTANT,value=0)
    else:
        # scale height, pad width
        result = resize(image, dsize=(0,0), fx=scale_height, fy=scale_height)
        padding = targetWidth - result.shape[1]
        p_left = int(padding/2)
        p_right = p_left if (padding%2) == 0 else p_left + 1
        assert padding >= 0 and (p_left + p_right) == padding, "unexpected width-padding: %d"%(padding)
        result = copyMakeBorder(result, top=0, bottom=0, left=p_left, right=p_right, borderType=BORDER_CONSTANT,value=0)                                    
    return result

####
# For a given file of commaseperated imagepaths and labels, load images.
# processImage: function to apply to image before resizing (in-place)
#
# channels: 
#    None: outputshape [n,h,w]
#    1: outputshape [n,h,w, 1]
#    3: outputshape [n,h,w, 3] value repeated thrice
####
def load_from_txt(txt_path, n_toLoad = None, seperators="[ ,]", resizeTo=None, keepRatio=False, imread_mode = IMREAD_GRAYSCALE, shuffleData=True, processImage=None, channels=None):
    images = []
    labels = []
    df = pd.read_csv(txt_path, sep=seperators ,header=None)   
    if shuffleData:
        df = df.sample(frac=1).reset_index(drop=True)
        
    n_rows = df.shape[0]
    if n_toLoad is not None:
        n_rows = n_toLoad
        
    for i in range( n_rows ):
        row = df.values[i]
        imagepath = str(Path(WMN_PATH) / row[0])
        label = row[1:]
        image = imread(imagepath, imread_mode)
        if not processImage is None:
            processImage(image)
        if not resizeTo is None:
            image = resize_image(image, resizeTo, keepRatio)
        images.append(image)
        labels.append(label)
        
      
    # add channel-dim
    images = np.array(images)
    if not channels is None:
        # add dim by reshaping
        shape = shape = [d for d in images.shape]
        shape.append(1)
        images = np.reshape(images, shape)
        if channels == 3:
            # repeat last dimension
            images = np.repeat(images, 3, axis=-1)

    return (images, np.array(labels).astype("int"))

def load_wmr_easy(n_toLoad = None, resizeTo=None, keepRatio=False, processImage=None, channels=None):
    txt_path = Path(WMN_PATH) / "easy_samples.txt"
    if n_toLoad is None:
        n_toLoad = 1000
    if n_toLoad > 1000:
        # load all 1000 images and duplicate available images randomly
        images_easy, labels_easy = load_from_txt(txt_path, n_toLoad=1000, seperators="[ ,]", resizeTo=resizeTo, keepRatio=keepRatio, processImage=processImage, channels=channels)
        n_diff = n_toLoad - len(images_easy)
        indices_toDuplicate = np.random.randint(0,len(images_easy), (n_diff))
        images_easy = np.append(images_easy, images_easy[indices_toDuplicate], 0)
        labels_easy = np.append(labels_easy, labels_easy[indices_toDuplicate], 0)
        print("attempted to load %d easy images while 1000 are available. Duplicating %d images." % (n_toLoad, n_diff))
    else:
        images_easy, labels_easy = load_from_txt(txt_path, n_toLoad=n_toLoad, seperators="[ ,]", resizeTo=resizeTo, keepRatio=keepRatio, processImage=processImage, channels=channels)        
    return (images_easy, labels_easy)

def load_wmr_diff_train(n_toLoad = None, resizeTo=None, keepRatio=False, processImage=None, channels=None):
    txt_path = Path(WMN_PATH) / "difficult_samples_for_train.txt"
    return load_from_txt(txt_path, n_toLoad=n_toLoad, seperators="[ ,]", resizeTo=resizeTo, keepRatio=keepRatio, processImage=processImage, channels=channels)
def load_wmr_diff_test(n_toLoad = None, resizeTo=None, keepRatio=False, processImage=None, channels=None):
    txt_path = Path(WMN_PATH) / "difficult_samples_for_test.txt"
    if n_toLoad is None:
        n_toLoad = 1000
    
    if n_toLoad > 1000:
        # load all 1000 images and duplicate available images randomly
        images, labels = load_from_txt(txt_path, n_toLoad=1000, seperators="[ ,]", resizeTo=resizeTo, keepRatio=keepRatio, processImage=processImage, channels=channels)
        n_diff = n_toLoad - len(images)
        indices_toDuplicate = np.random.randint(0,len(images), (n_diff))
        images = np.append(images, images[indices_toDuplicate], 0)
        labels = np.append(labels, labels[indices_toDuplicate], 0)
        print("attempted to load %d diff-test images while 1000 are available. Duplicating %d images." % (n_toLoad, n_diff))
    else:
        images, labels = load_from_txt(txt_path, n_toLoad=n_toLoad, seperators="[ ,]", resizeTo=resizeTo, keepRatio=keepRatio, processImage=processImage, channels=channels) 
    
    return (images, labels)
    


def load_wmr_easy_split(n_toLoad = None, resizeTo=None, keepRatio=False, processImage=None, channels=None):
    txt_train = Path(WMN_PATH) / "easy_samples_train.txt"
    txt_test = Path(WMN_PATH) / "easy_samples_test.txt"
    
    if not txt_train.exists():
        split_wmr_easy()
    
    data_train = load_from_txt(txt_train, n_toLoad=n_toLoad, seperators="[ ,]", resizeTo=resizeTo, keepRatio=keepRatio, processImage=processImage, channels=channels)
    data_test = load_from_txt(txt_test, n_toLoad=n_toLoad, seperators="[ ,]", resizeTo=resizeTo, keepRatio=keepRatio, processImage=processImage, channels=channels)
    return (data_train, data_test)



# loads "easy_samples.txt", splits rows equally into "easy_samples_train.txt" and "easy_samples_test.txt" 
def split_wmr_easy():
    dataset_path = Path(WMN_PATH) 
    txt_easy = dataset_path / "easy_samples.txt"
    txt_train = dataset_path / "easy_samples_train.txt"
    txt_test = dataset_path / "easy_samples_test.txt"
    seperators = "[ ,]"
    # open file
    df = pd.read_csv(txt_easy, sep=seperators ,header=None)   
    # shuffle dataset
    df = df.sample(frac=1).reset_index(drop=True)
    df_size = len(df.index)
    assert df_size % 2 == 0, "cant split dataset, len uneven"
    split_size = df_size // 2
    # split and save
    df_train = df.iloc[0:split_size]
    df_test = df.iloc[split_size:None]
    df_train.to_csv(txt_train, sep="," ,header=None, index=False)
    df_test.to_csv(txt_test, sep="," ,header=None, index=False)
    
# select n random images from difficult_samples_for_train.txt,
# write them into diff_train_split_n.txt 
def split_wmr_diff_train(n=500):
    dataset_path = Path(WMN_PATH) 
    # txt-file to load
    txt_path = Path(WMN_PATH) / "difficult_samples_for_train.txt"
    # txt-file to create
    filename = "diff_train_split_%d.txt" % (n)
    newFile_path = Path(WMN_PATH) / filename
    assert not newFile_path.exists(), "txt already exists"
    seperators = "[ ,]"
    
    # open file
    df = pd.read_csv(txt_path, sep=seperators ,header=None)   
    # shuffle dataset
    df = df.sample(frac=1).reset_index(drop=True)
    df_size = len(df.index)
    assert df_size >= n, "split longer than dataset"
    
    df_split = df.iloc[0:n]
    df_split.to_csv(newFile_path, sep="," ,header=None, index=False)

# loads images given by diff_train_split_n.txt,
# and if that file does not exist, calls split_wmr_diff_train(n)
def load_diff_split(n_toLoad = None, resizeTo=None, keepRatio=False, processImage=None, channels=None, n=500):
    dataset_path = Path(WMN_PATH) 
    filename = "diff_train_split_%d.txt" % (n)
    # txt-file to load
    txt_path = Path(WMN_PATH) / filename
    if not txt_path.exists():
        split_wmr_diff_train(n)
    assert txt_path.exists(), "diff-split does not exist yet."
    data = load_from_txt(txt_path, n_toLoad=n_toLoad, seperators="[ ,]", resizeTo=resizeTo, keepRatio=keepRatio, processImage=processImage, channels=channels)
    return data
    
    


# args:
#    datasetPath - <pathlib.Path>: path to dataset
#    n_images - <int>: number of images to load.
#                      if -1, load every image
#    imread_mode-<enum>: how images are to be opened. Default color(3-channels)
#    shuffle - <bool>: whether to shuffle images
#    resize_to <list[2]>: dimenisions to resize images to, if given. (width, height)
####
# returns:
#    list of len 2: loaded images, paths of images in same order
def load_wmr(datasetPath, n_images=-1, imread_mode=IMREAD_GRAYSCALE, shuffleImages=True, resize_to=None):
    imagePaths = [
        str(datasetPath / imageName) for imageName in listdir(datasetPath)
    ]
    if shuffleImages:
        shuffle(imagePaths)
    if n_images > 0:
        imagePaths = imagePaths[0:n_images]
    images = [
        imread(imagepath, imread_mode) for imagepath in imagePaths
    ]
    if resize_to != None:
        images = [
            resize(image, resize_to) for image in images
        ]
    
    return (images, imagePaths)
    