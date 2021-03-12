import numpy as np
import keras
from keras import backend as K
import tensorflow as tf
from keras.applications.vgg16 import VGG16

def TV_loss(y_pred):
    loss = K.mean(K.abs(y_pred[:,1:,:,:]-y_pred[:,:-1,:,:]))
    loss = loss +K.mean(K.abs(y_pred[:,:,1:,:]-y_pred[:,:,:-1,:]))
    return loss


def L1_loss(y_true,y_pred):
    L1_loss = K.mean(K.abs(y_true-y_pred),axis=-1)
    return L1_loss


def custom_loss(y_true,y_pred):
    L1_loss = K.mean(K.abs(y_true-y_pred),axis=-1)
#    TV_loss = K.mean(K.abs(y_pred[:,1:,:,:]-y_pred[:,:-1,:,:]))
#    TV_loss = TV_loss +K.mean(K.abs(y_pred[:,:,1:,:]-y_pred[:,:,:-1,:]))
    SSIM_loss =  tf.reduce_mean(tf.image.ssim_multiscale(y_true,y_pred,max_val=1.0))
    loss = L1_loss + 5 * (1.0-SSIM_loss)
    return loss


def SSIM(y_true,y_pred):
    SSIM_loss = tf.reduce_mean(tf.image.ssim_multiscale(y_true,y_pred,max_val=2.0,filter_size=13))
    return SSIM_loss


def TumorSensitiveLoss(y_true,y_pred):
    diff = K.abs(y_true-y_pred)
    patch = keras.layers.AveragePooling2D(pool_size=(9, 9), strides=None, padding='valid')(diff)
    patch = K.maximum(0.0,patch-3.0*K.mean(patch))
    loss = 100000* K.mean(patch**3) + K.mean(diff)
    return loss


def VGGloss(y_true, y_pred):
    vggmodel = VGG16(include_top=False,weights='imagenet')
    for layer in vggmodel.layers:
        layer.trainable = False
    pred = K.concatenate([y_pred, y_pred, y_pred])
    true = K.concatenate([y_true, y_true, y_true])
    f_p = vggmodel(pred)
    f_t = vggmodel(true)
    VGG_loss = K.mean(K.square(f_p-f_t))
    L1_loss = K.mean(K.square(y_true-y_pred),axis=-1)
    SSIM_loss =  tf.reduce_mean(tf.image.ssim_multiscale(y_true,y_pred,max_val=1.0))
    loss = 1.0 * L1_loss + 1.0 * (1-SSIM_loss) + 0.0 * VGG_loss
    return loss



def perceptual_loss(y_true, y_pred):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))



def Data_shuffle(x_train,y_train):
    ind = np.arange(x_train.shape[0])
    np.random.shuffle(ind)
    x_train = x_train[ind,:,:,:]
    y_train = y_train[ind,:,:,:]
    return x_train, y_train


def OverSample(x_tumor,y_tumor,times):
    for i in range(times):
        x_tumor = np.concatenate((x_tumor,x_tumor),axis=0)
        y_tumor = np.concatenate((y_tumor,y_tumor),axis=0)
    return (x_tumor,y_tumor)


