from tensorflow.keras.layers import Input, Conv2D
from cyclegan.generator import generator as generator_modified
from cyclegan.generator import generator_small as gen_small
from cyclegan.discriminator import discriminator as discriminator_modified
from cyclegan.discriminator import discriminator_small

####
# returns a generator.
####
def generator(image_shape, norm_type='instancenorm', smallModel=False):
    if smallModel:
        return gen_small(image_shape, norm_type=norm_type)
    else:
        return generator_modified(image_shape, norm_type=norm_type)
####
# returns a discriminator.
####
def discriminator(n_channels, norm_type='instancenorm', smallModel=False):
    if smallModel:
        return discriminator_small(n_channels, norm_type=norm_type)
    else:
        return discriminator_modified(n_channels, norm_type=norm_type)
























