# add root dir to syspath
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#parent_dir = os.path.dirname(current_dir)
#root_dir = os.path.dirname(parent_dir)
sys.path.insert(0, current_dir)


from A_FullyConvolutionalNet import convNet
from B_TemporalMapper import temporalMapper

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.backend import ctc_batch_cost, ctc_decode
import time
from pathlib import Path

saved_models_path =  Path(__file__).parent.absolute() / "saved_models"

####
#  FULLY CONVOLUTIONAL SEQUENCE RECOGNITION NETWORK
####
# https://ieeexplore.ieee.org/document/8606091
####
class fcsrn():
    
    def __init__(self, input_shape, checkpoint_path = None, epoch_to_load=None):
        self.input_shape = input_shape
        
        if checkpoint_path is not None:          
            checkpoint_path = saved_models_path/checkpoint_path
        self.checkpoint_path = checkpoint_path
        
        # build model
        input, convNet_output = convNet(input_shape)
        output = temporalMapper(convNet_output)
        output = Activation("softmax")(output)
        self.model = tf.keras.Model(inputs=input, outputs=output)
        
        self.optimizer = SGD(lr=0.01, momentum=0.9, decay=0.0001)
        
        # prepare checkpoint
        if checkpoint_path != None:
            # if folder does not exist, create it
            if not self.checkpoint_path.exists():
                self.checkpoint_path.mkdir()
            # folder for saving Accuracy
            self.ar_folder = checkpoint_path / "ar"
            if not self.ar_folder.exists():
                self.ar_folder.mkdir()
            # init checkoint    
            self.checkpoint = tf.train.Checkpoint(
                model = self.model,
                optimizer = self.optimizer,
            )
            self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, checkpoint_path, max_to_keep=None, checkpoint_name="epoch")
            # load existing checkpoint if it exists
            if self.checkpoint_manager.latest_checkpoint:
                if epoch_to_load is None:                    
                    checkpoint_to_be_loaded = self.checkpoint_manager.latest_checkpoint
                else:
                    checkpoint_to_be_loaded = str(checkpoint_path / ("epoch-%d" % (epoch_to_load)))
                self.checkpoint.restore(checkpoint_to_be_loaded)
                print("loaded checkpoint: ", checkpoint_to_be_loaded)
            else:                
                print("created new Model")
        else:
            self.checkpoint = None
            print("No checkpointpath given, model will not be saved.")
            
    ####
    # ctc-loss as lossfunction.
    # for now, without Aug-ctc-Loss
    ####
    def lossfunction(self, modeloutput, target, batchsize):
        y_true = target
        y_pred = modeloutput
        n_time_slices = self.input_shape[1]/8 # fcsrn-model halves inputdimenions 3 times -> finalwidth = inputwidth / 8
        input_length = np.full((batchsize, 1), n_time_slices)
        label_length = np.full((batchsize, 1), 5)
        loss = ctc_batch_cost(y_true, y_pred, input_length, label_length)       
        return loss
    
    @tf.function
    def train_step(self, inputImages, targetLabels, batchsize):
        with tf.GradientTape(persistent=True) as tape:
            modelOutput = self.model(inputImages, training=True)
            loss = self.lossfunction(modelOutput, targetLabels, batchsize)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    
    
    
    ####
    # train_X, train_Y: datasets of images and labels for training
    # epochs, batchsize: Trainingparams
    # testSet: triplets (name, testimages, testlabels) for which to calculate AR
    # 
    ####  
    def train(self, train_X, train_y, epochs, batchsize, testSet=None, checkpointInterval=10, verbose=True):
        trainstart = time.time()
        n_batches = tf.data.experimental.cardinality(train_X).numpy()
        # iterate epochs
        for epoch in range(1, epochs + 1):
            epochstart = time.time()
            step = 0
            if verbose:
                progBar = tf.keras.utils.Progbar(n_batches)
            # iterate batches
            for inputBatch, targetBatch in tf.data.Dataset.zip((train_X, train_y)):
                self.train_step(inputBatch, targetBatch, batchsize)
                if verbose:
                    progBar.add(1)                
                step += 1
            ###
            # epoch finished
            # output epochtime
            if verbose:
                print("epoch %d took: %.2f seconds" % (epoch, time.time() - epochstart))
            # calculate current character accuracy rate
                self.log_ar(testSet, epoch)
            # save checkpoint at intervals
            if (not checkpointInterval is None) and (epoch % checkpointInterval) == 0:
                savepath = self.checkpoint_manager.save(checkpoint_number=epoch)
                print("saved to: {}".format(savepath))
                
                
        # training finished
        if verbose:
            print("training took %.2f seconds" % (time.time() - trainstart) )
    ####
    # For every triplet (name, testimages, testlabels),
    # in <testSet>,
    # calculates Character-Accuracy-Rate (AR) of current model and appends
    # it to <name>.csv
    ####
    def log_ar(self, testSet, epoch):
        if self.ar_folder is None or testSet is None:
            return
        
        for name, images_test, labels_test in testSet:
            filename = name + ".csv"
            filepath = self.ar_folder / filename 
            if not filepath.exists():
                filepath.touch()
            
            # calculate ar
            stime = time.time()
            ar = self.character_accuracyRate(images_test.astype("float32"), labels_test)
            print("ar_%s(epoch=%d)=%f\t%fseconds" % (name, epoch, ar, time.time() - stime) )
            
            with filepath.open("a") as f:
                f.write("%d,%f\n" % (epoch, ar) )
          
    ####
    # Predicts labels from inputimages
    ####
    @tf.function
    def decode(self, inputImages, batchsize):
        modelOutput = self.model(inputImages, training=False)
        time_slices = self.input_shape[1]/8 
        input_length = np.full((batchsize), time_slices)
        decoded, probs = ctc_decode(modelOutput, input_length, greedy=True)
        labels =  tf.dtypes.cast(decoded[0], tf.int32)
        return labels
     
    ####
    # Transform dense tensor to sparse.
    ####
    def labelToSparse(self, dense, remove_blanks=True):
        if remove_blanks:
            indices = tf.where(tf.not_equal(dense, -1))
        else:
            indices = tf.where(tf.equal(dense, dense))
        result = tf.SparseTensor(indices, tf.gather_nd(dense, indices), tf.shape(dense, out_type=tf.int64))
        return result
    
    
    
    ####
    # calculate elementwise editdistances for two sets of labels
    ####
    @tf.function
    def labelDistance(self, pred, truth):
        # convert dense tensors to sparse.
        pred_s = self.labelToSparse(pred)        
        truth_s = self.labelToSparse(truth)        
        distance = tf.edit_distance(pred_s, truth_s, normalize=False)        
        return tf.cast(distance, tf.int32)
    
    @tf.function
    def character_accuracyRate(self, images_test, labels_test):           
        # predict labels from inputimages
        labels_result = self.decode(images_test, len(images_test))
        # count number of blanks (-1) in predicted labels
        #blanks = len(tf.where(tf.equal(labels_result, -1)))
        # calculate editdistance of each labelpair
        distances = self.labelDistance(labels_result, labels_test)
        # sum distances, subtract number of blanks
        sumDistances = tf.math.reduce_sum(distances)# - blanks
        # get number of charactes
        n_characters = tf.size(labels_test)
        # calculate character accuracy rate
        ar = 1 - (sumDistances / n_characters) 
        return ar    
               