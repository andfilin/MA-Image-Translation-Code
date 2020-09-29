from enum import Enum
####
# enum of parameternames to log.
####
class parameters(Enum):
    model = "model"
    n_images = "n_images"
    imageshape = "imageshape"
    batchsize = "batchsize"
    _lambda = "lambda"
    lr = "lr"
    adversial_lossfunction= "adversial_lossfunction"
    poolsize = "poolsize"
    epochs_trained = "epochs_trained"
    trainingsessions = "trainingsessions"
    smallModel = "smallModel"