from keras import backend as K
import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import multi_gpu_model
from keras.callbacks import ReduceLROnPlateau
import h5py as h5
import keras
from keras.models import load_model
from unet_3D import *
from utils import *
from DataAugment import *
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,4"
os.environ['OMP_NUM_THREADS']='10'
batch_size = 3
epochs = 4

'''load data'''
#MRIdata = h5.File('/raid/chenchao/code/brainMRI/mask/MRIData.h5','r')
#TrainData = MRIdata['TrainData']
#TrainMask = MRIdata['TtainMask']
#TestData = MRIdata['TestData']
#TestMask = MRIdata['TestMask']
#TrainData = np.swapaxes(TrainData,1,3)
#TrainMask = np.swapaxes(TrainMask,1,3)
#TestData = np.swapaxes(TestData,1,3)
#TestMask = np.swapaxes(TestMask,1,3)

MRIData = h5.File('/raid/chenchao/code/brainMRI/mask/MRIData_subset_3D.h5','r')
x_train = MRIData['x_train']
y_train = MRIData['y_train']
x_test = MRIData['x_test']
y_test = MRIData['y_test']
x_train = np.array(x_train)
y_train = np.expand_dims(y_train,axis=-1)
x_test = np.array(x_test)
y_test = np.expand_dims(y_test,axis=-1)

start = 35
end = 45
#x_train = x_train[:,start:end,:,:,:]
#y_train = y_train[:,start:end,:,:,0]
#y_train = np.expand_dims(y_train,axis=-1)
print ('x_train shape:', x_train.shape)
#print ('y_train shape:', y_train.shape)

#x_test = x_test[:,start:end,:,:,:]
#y_test = y_test[:,start:end,:,:,0]
#y_test = np.expand_dims(y_test,axis=-1)
print ('x_test shape:', x_test.shape)
print ('y_test shape:', y_test.shape)


########################### Model Construction ######################
model = unet(batch_size = batch_size, depth=80, height = 256, width = 192, channel = 3, classes =1)
model.summary()
Model = multi_gpu_model(model,gpus=3)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='auto', factor=0.2, patience=6, min_lr=0.00000001)
Adam = keras.optimizers.Adam(lr=0.0001,beta_1=0.9,beta_2=0.999)
Model.compile(optimizer=Adam, loss=custom_loss, metrics=['MAE'])
#model.load_weights('model_3D_weights.h5')
#for num, layer in enumerate(model.layers[:-30]):
#    layer.trainable = True
#    print (num,layer.trainable)
history = Model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test,y_test), callbacks=[reduce_lr])
score = Model.evaluate(x_test,y_test,batch_size=batch_size)
print ('Test Loss:', score[0])
print ('Test MSE:', score[1])



model.save('model_3D.h5')
model.save_weights('model_3D_weights.h5')

print ('Done')
