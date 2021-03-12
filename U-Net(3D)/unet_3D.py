import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from tensorflow.image import adjust_gamma



def SliceTensor(index):
    def func(x):
        return x[:,:,:,:,index]
    return Lambda(func)



def ConvP3D(x_in, out_filters, strides=(1,1,1)):
    x = Conv3D(out_filters, kernel_size=(1,3,3), padding='same', strides=strides, activation='relu', use_bias=True, kernel_initializer='he_normal')(x_in)
    x = Conv3D(out_filters, kernel_size=(3,1,1), padding='same', strides=(1,1,1), activation='relu', use_bias=True, kernel_initializer='he_normal')(x)
#    x = Conv3D(out_filters, kernel_size=(1,3,3), padding='same', strides=(1,1,1), activation='relu', use_bias=True, kernel_initializer='he_normal')(x)
    return x


def unet(batch_size, depth, height, width, channel, classes):
    inputs = Input(shape=(depth,height,width,channel))
    conv1 = ConvP3D(inputs,64)
    conv1 = ConvP3D(conv1,64)
#    conv1 = ConvP3D(conv1,64)

    pool1 = MaxPooling3D(pool_size=(2,2,2))(conv1)
    conv2 = ConvP3D(pool1,128)
    conv2 = ConvP3D(conv2,128)
#    conv2 = ConvP3D(conv2,128)

    pool2 = MaxPooling3D(pool_size=(2,2,2))(conv2)
    conv3 = ConvP3D(pool2,256)
    conv3 = ConvP3D(conv3,256)
#    conv3 = ConvP3D(conv3,256)

    pool3 = MaxPooling3D(pool_size=(2,2,2))(conv3)
    conv4 = ConvP3D(pool3,512)
    conv4 = ConvP3D(conv4,512)
#    conv4 = ConvP3D(conv4,512)


    drop4 = Dropout(1.0)(conv4)
    pool4 = MaxPooling3D(pool_size=(1,2,2))(drop4)

    conv5 = ConvP3D(pool4,1024)
    conv5 = ConvP3D(conv5,1024)
    conv5 = ConvP3D(conv5,1024)
    conv5 = non_local_block(conv5, intermediate_dim = None, compression =2, mode='embedded', add_residual = True)



    drop5 = Dropout(1.0)(conv5)

    up6 = Conv3D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (1,2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 4)
    conv6 = ConvP3D(merge6,512)
    conv6 = ConvP3D(conv6,512)
    conv6 = ConvP3D(conv6,512)
    conv6 = non_local_block(conv6, intermediate_dim = None, compression =2, mode='embedded', add_residual = True)



    up7 = Conv3D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 4)
    conv7 = ConvP3D(merge7,256)
    conv7 = ConvP3D(conv7,256)
    conv7 = ConvP3D(conv7,256)



    up8 = Conv3D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 4)
    conv8 = ConvP3D(merge8,128)
    conv8 = ConvP3D(conv8,128)
    conv8 = ConvP3D(conv8,128)



    up9 = Conv3D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 4)
    conv9 = ConvP3D(merge9,64)
    conv9 = ConvP3D(conv9,64)
    conv9 = ConvP3D(conv9,64)


    conv9 = Conv3D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv3D(1, 1, activation = 'relu')(conv9)

    T1 = SliceTensor(0)(inputs)
    T1 = Lambda(lambda x: K.expand_dims(x,-1))(T1)
    T2 = SliceTensor(1)(inputs)
    T2 = Lambda(lambda x: K.expand_dims(x,-1))(T2)
    out = multiply([T1,conv10])

    model = Model(input = inputs, output = out)
    return model


