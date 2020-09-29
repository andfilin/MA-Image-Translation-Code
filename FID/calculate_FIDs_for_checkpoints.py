from cyclegan.cyclegan import cyclegan
from dataset import load_realdata
from imageGenerators.imgGen_simple import synth_generator
from imageGenerators.imgGen_augmented import synth_generator as new_generator
import FID_interface

import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE
import re
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

saved_models_path =  Path(__file__).parent.parent.absolute() / "cyclegan/saved_models"

###
# calculate FID for generated images of each epoch of a model.
###
# model_checkpointPath: Where Checkpoints are saved
# smallModel<Bool>:     Whether to use smaller Model
# n_images:             How many images to use for calculating FID
# input_synthethic:     whether the input to cyclegan are synthetic images.
#                            If false, use real easy images as input. 
# singleEpoch           if True, calculate FID only for epoch <epochstart>
def calculate_FIDs(model, smallModel, n_images, stepsize = 1, n_digits=5, epochstart = 0, input_synthethic=True, useNewGenerator=True, singleEpoch=False):
    model_checkpointPath = saved_models_path/model
    # get inputdimensions of cyclegan
    inputshapePath = model_checkpointPath / "inputshape"
    cyclegan_shape = [int(s) for s in inputshapePath.read_text().split(",")]
    input_height = cyclegan_shape[0]
    input_width = cyclegan_shape[1]
    
    ###
    # get list of epochInts
    ###
    # list of checkpointpaths
    paths = [path for path in model_checkpointPath.glob("*.index")]
    # extract filenames from paths
    names = [
        re.search(r"epoch-\d+\.index", str(path)).group() for path in paths
    ]
    # extract epochcounts (int) from names
    epoch_ints = [
        int(re.search(r"\d+", name).group()) for name in names
    ]
    epoch_ints = sorted(epoch_ints)
    
    # start from Epoch <epochstart>
    epoch_ints = epoch_ints[epoch_ints.index(epochstart):None]
    if singleEpoch:
        epoch_ints = epoch_ints[0:1]
        assert len(epoch_ints) == 1 and epoch_ints[0] == epochstart
    
    ###
    # init file to write fid-values to
    fid_file = model_checkpointPath / "FID.txt"
    fid_csv = model_checkpointPath / "FID.csv"
    fid_csv_diff = model_checkpointPath / "FID_diff.csv"
    if not fid_file.exists():
        fid_file.touch()
        fid_file.write_text("\tFIDs for n_images = %d\n" % (n_images) )
    if not fid_csv.exists():
        fid_csv.touch()    
    if not fid_csv_diff.exists():
        fid_csv_diff.touch() 
    
    ###
    # load easy/difficult real samples and calc their fid-stats
    images_easy, _ = load_realdata.load_wmr_easy(n_images, (input_width, input_height), channels=1)
    images_diff_test, _ = load_realdata.load_wmr_diff_test(n_images, (input_width, input_height), channels=1)
    easy_stats = FID_interface.calculate_stats(images_easy, printTime=True)
    difficult_stats = FID_interface.calculate_stats(images_diff_test, printTime=True)
    #del images_easy
    del images_diff_test
    
    cg_input = None
    
    # as input for cgmodel, take either synthetic images or real easy images
    if input_synthethic:
        # generate synthetic images
        labels = np.random.randint(0,20,(n_images, n_digits))
        images_synthetic = generate_synthethic(labels, n_digits, input_width, input_height, useNewGenerator)
        # if model trained on 3-channel images, repeat last channel
        if len(cyclegan_shape) == 3 and cyclegan_shape[2] == 3:
            shape = list(images_synthetic.shape)
            if len(shape) == 3:
                shape.append(1)
                images_synthetic = np.reshape(images_synthetic, shape)
            images_synthetic = np.repeat(images_synthetic, 3, axis=-1)
            assert images_synthetic.shape[-1] == 3
        cg_input = images_synthetic    
    else:
        cg_input = images_easy[0:n_images]
        
    print("inputimages:")
    print(cg_input.shape)
    print(cg_input.dtype)
    for i in range(5):
        image = cg_input[i]
        plt.imshow(image[:,:,0], cmap="gray", vmin=0, vmax=255)
        plt.show()
    
    ###
    # for every epoch calculate and write FID
    for epoch in epoch_ints:
        # load model
        cgModel = cyclegan(cyclegan_shape,0,1, "mse",0,0,  checkpoint_path=model_checkpointPath, load_checkpoint_after_epoch=epoch, smallModel=smallModel)
        
        # translate images
        images_translated = translate_images(cg_input, cgModel)
        del cgModel
        # calc fid-stats for translated images
        translated_stats = FID_interface.calculate_stats(images_translated, printTime=False)
        # show translated images
        for i in range(5):
            image = images_translated[i,:,:,0]
            plt.imshow(image, cmap="gray")
            plt.show() 
        del images_translated
        # get fid-values        
        fid_easy = FID_interface.calculate_fid_from_stats(translated_stats, easy_stats)
        fid_difficult = FID_interface.calculate_fid_from_stats(translated_stats, difficult_stats)
        
        # append to file
        text = "--------------------\n"
        text += "Epoch %d:\n" % (epoch)
        text += "fid(gen, wmn_easy) =\t%f\n" % (fid_easy)
        text += "fid(gen, wmn_difficult) =\t%f\n" % (fid_difficult)
        print(text)
        
        # do not save values when only examining single epoch
        if singleEpoch:
            continue
        
        with fid_file.open("a") as f:
            f.write(text)
        text_csv = "%d,%f\n" % (epoch, fid_easy)
        with fid_csv.open("a") as f:
            f.write(text_csv)
        
        # csv diff images
        text_csv_diff = "%d,%f\n" % (epoch, fid_difficult)
        with fid_csv_diff.open("a") as f:
            f.write(text_csv_diff)
            
####
# given 2 models:
#     cyclegan_1, which translates synthethic images to real-easy images;
#     cyclegan_2, which translates (real) easy images to real-difficult images
# For a specific epoch for cyclegan_1, calculates for each epoch of cyclegan_2 
#     FID between cyclegan_2( cyclegan_1(synthethic_images) ) and real-difficult images.
####
def calculate_FID_2Models(modelPath_A, modelPath_B, modelA_epoch, n_images, epochstart=0, modelA_isSmall=True, modelB_isSmall=True, n_digits=5, singleEpoch=False):
    # get inputdimensions of cyclegan - assume both model use same shape
    inputshapePath = modelPath_A / "inputshape"
    cyclegan_shape = [int(s) for s in inputshapePath.read_text().split(",")]
    input_height = cyclegan_shape[0]
    input_width = cyclegan_shape[1]
    
    ###
    # get list of epochInts
    ###
    # list of checkpointpaths
    paths = [path for path in modelPath_B.glob("*.index")]
    # extract filenames from paths
    names = [
        re.search(r"epoch-\d+\.index", str(path)).group() for path in paths
    ]
    # extract epochcounts (int) from names
    epoch_ints = [
        int(re.search(r"\d+", name).group()) for name in names
    ]
    epoch_ints = sorted(epoch_ints)
    
    # start from Epoch <epochstart>
    epoch_ints = epoch_ints[epoch_ints.index(epochstart):None]
    if singleEpoch:
        epoch_ints = epoch_ints[0:1]
        assert len(epoch_ints) == 1 and epoch_ints[0] == epochstart
    
    ###
    # init file to write fid-values to
    fid_file = modelPath_B / "FID.txt"
    csv_easy = modelPath_B / "FID_easy.csv"
    csv_diff = modelPath_B / "FID_diff.csv"
    if not fid_file.exists():
        fid_file.touch()
        fid_file.write_text("\tFIDs for n_images = %d\n" % (n_images) )
    if not csv_easy.exists():
        csv_easy.touch()    
    if not csv_diff.exists():
        csv_diff.touch()
        
    ###
    # load easy/difficult real samples and calc their fid-stats
    images_easy, _ = load_realdata.load_wmr_easy(n_images, (input_width, input_height), channels=1)
    images_diff_test, _ = load_realdata.load_wmr_diff_test(n_images, (input_width, input_height), channels=1)
    easy_stats = FID_interface.calculate_stats(images_easy, printTime=True)
    difficult_stats = FID_interface.calculate_stats(images_diff_test, printTime=True)
    del images_easy
    del images_diff_test
    
    ###
    # translate synthethic images to easy-images
    labels = np.random.randint(0,20,(n_images, n_digits))
    images_synthetic = generate_synthethic(labels, n_digits, input_width, input_height, useNewGenerator=True)
    # translate images
    model1 = cyclegan(cyclegan_shape,0,1, "mse",0,0,  checkpoint_path=modelPath_A, load_checkpoint_after_epoch=modelA_epoch, smallModel=modelA_isSmall)
    translated_easy = translate_images(images_synthetic, model1)
    del model1
    
    ####
    # iterate epochs of model 2, translate translated_easy to translated_diff
    for epoch in epoch_ints:
        # load model
        model2 = cyclegan(cyclegan_shape,0,1, "mse",0,0,  checkpoint_path=modelPath_B, load_checkpoint_after_epoch=epoch, smallModel=modelB_isSmall)
        # translate
        translated_diff = translate_images(translated_easy, model2)
        del model2
        # show translated images
        for i in range(5):
            image = translated_diff[i,:,:,0]
            plt.imshow(image, cmap="gray", vmin=0, vmax=255)
            plt.show()   
        # calc FID
        result_stats = FID_interface.calculate_stats(translated_diff, printTime=False)
        del translated_diff
        fid_easy = FID_interface.calculate_fid_from_stats(result_stats, easy_stats)
        fid_difficult = FID_interface.calculate_fid_from_stats(result_stats, difficult_stats)
        
        # txt
        text = "--------------------\n"
        text += "Epoch %d:\n" % (epoch)
        text += "fid(gen, wmn_easy) =\t%f\n" % (fid_easy)
        text += "fid(gen, wmn_difficult) =\t%f\n" % (fid_difficult)
        print(text)
        
        # do not save values when only examining single epoch
        if singleEpoch:
            continue
        
        ##
        # append to files
        ##
        # csv
        text_easy = "%d,%f\n" % (epoch, fid_easy)
        text_diff = "%d,%f\n" % (epoch, fid_difficult)
        
        with csv_easy.open("a") as f:
            f.write(text_easy)
        with csv_diff.open("a") as f:
            f.write(text_diff)            
        
        with fid_file.open("a") as f:
            f.write(text)
        
        
        
def generate_synthethic(labels, n_digits=5, input_width=0, input_height=0, useNewGenerator=False):
    if useNewGenerator:
        synthGenerator = new_generator()
        imageDims = (input_width, input_height)
        images_synthetic = synthGenerator.makeImages(labels, resizeTo=imageDims, color=True, rotate=True)
        return images_synthetic
     
    # using old generator
    #
    # margins between digits, padding around resultimage 
    margins = [30 for _ in range(0, n_digits - 1)]
    padding = (0,0, 1,1) # top,bottom, left, right
    # margin between digits in same column
    vertical_margin = 20
    ####
    # ranges for how far digits can scroll up or down, 0 meaning no scrolling, 0.5 meaning halfway to next digit
    range_normal=(-0.1,0.1)
    range_midstate=(0.3,0.7)
    
    synthGen = synth_generator(vertical_margin=vertical_margin)    
    images_synthetic = synthGen.generate_images(labels, margins, padding, input_width, input_height, draw_vertical_seperators=False, range_normal=range_normal,range_midstate=range_midstate)
    
    return images_synthetic
    
    
    
    
####
# normalize inputimages from [0,255] to [-1,1],
# translate using model, 
# denormalize and return result
####
def translate_images(synthImages, i2i_model):
    model_input = tf.data.Dataset.from_tensor_slices(synthImages)\
                .map(i2i_model.preprocess_input, num_parallel_calls=AUTOTUNE)\
                .cache()\
                .batch(1)
    ####
    # predict realistic images
    translated_images = i2i_model.gen_AtoB.predict(model_input)
    # denormalize
    translated_images = (translated_images + 1) * 127.5
    return translated_images
    