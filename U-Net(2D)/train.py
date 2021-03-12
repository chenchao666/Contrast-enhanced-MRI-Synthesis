from keras import backend as K
import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import multi_gpu_model
from keras.callbacks import ReduceLROnPlateau
import h5py as h5
import keras
from keras.models import load_model
from unet import *
from utils import *
from DataAugment import *
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
os.environ['OMP_NUM_THREADS']='10'
batch_size = 64
epochs = 50

'''load data'''
MRIdata = h5.File('/raid/chenchao/code/brainMRI/mask/MRIData_subset.h5','r')
x_train = MRIdata['x_train']
y_train = MRIdata['y_train']
x_test = MRIdata['x_test']
y_test = MRIdata['y_test']
x_train = np.array(x_train)
y_train = np.array(y_train)


#TrainData = MRIdata['TrainData']
#TrainMask = MRIdata['TtainMask']
#TestData = MRIdata['TestData']
#TestMask = MRIdata['TestMask']
#TrainData = np.swapaxes(TrainData,1,3)
#TrainMask = np.swapaxes(TrainMask,1,3)
#TestData = np.swapaxes(TestData,1,3)
#TestMask = np.swapaxes(TestMask,1,3)
#start = 60
#end = 140
#x_train = TrainData[:,start:end,:,:,:]
#y_train = TrainMask[:,start:end,:,:,0]

#x_train = np.reshape(x_train,(x_train.shape[0]*x_train.shape[1],x_train.shape[2],x_train.shape[3],3))
#y_train = np.reshape(y_train,(y_train.shape[0]*y_train.shape[1],y_train.shape[2],y_train.shape[3],1))
print ('x_train shape:', x_train.shape)
print ('y_train shape:', y_train.shape)

#x_test = TestData[:,start:end,:,:,:]
#y_test = TestMask[:,start:end,:,:,0]
#x_test = np.reshape(x_test,(x_test.shape[0]*x_test.shape[1],x_test.shape[2],x_test.shape[3],3))
#y_test = np.reshape(y_test,(y_test.shape[0]*y_test.shape[1],y_test.shape[2],y_test.shape[3],1))
print ('x_test shape:', x_test.shape)
print ('y_test shape:', y_test.shape)

#x_tumor = np.load('tumorData1.npy')
#y_tumor = np.load('tumorMask1.npy')
#x_tumor = 0.5*(x_tumor+1)
#y_tumor = 0.5*(y_tumor+1)
#print (y_tumor.shape)
#y_tumor = y_tumor[:,:,:,0]
#y_tumor = np.expand_dims(y_tumor,axis=3)

#x_tumor,y_tumor = OverSample(x_tumor,y_tumor,times=3)
#x_train = np.concatenate((x_train,x_tumor),axis=0)
#y_train = np.concatenate((y_train,y_tumor),axis=0)
#x_train,y_train = CrossMixup(x_train,y_train,x_tumor,y_tumor,num=80000)
#x_train,y_train = Data_shuffle(x_train,y_train)


########################### Model Construction ######################
model = unet(batch_size = batch_size, height = 256, width = 192, channel = 3, classes =1)
#Model.summary()
Model = multi_gpu_model(model,gpus=2)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='auto', factor=0.2, patience=5, min_lr=0.00000001)
Adam = keras.optimizers.Adam(lr=0.0001,beta_1=0.9,beta_2=0.999)
Model.compile(optimizer=Adam, loss=custom_loss, metrics=['MAE'])
#model.load_weights('model_unet.h5')
#for num, layer in enumerate(model.layers[:-5]):
#    layer.trainable = False
#    print (num,layer.trainable)
history = Model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test,y_test), callbacks=[reduce_lr])
score = Model.evaluate(x_test,y_test,batch_size=batch_size)
print ('Test Loss:', score[0])
print ('Test MAE:', score[1])

model.save('model_unet.h5')
model.save_weights('model_weights_unet.h5')


####################################################
TestImg = TestData[:,90,:,:,:]
GroundTruth = TestMask[:,90,:,:,0]
T1 = TestData[:,90,:,:,0]
GroundTruth = 255*GroundTruth
T1 = 255*T1
#layer=K.function([Model.layers[0].input],[Model.layers[-1].output])
#Prediction=layer([TestImg])[0]
Prediction = Model.predict(TestImg)
print (Prediction.shape)
print (GroundTruth.shape)
Prediction = 255*Prediction
for i in range(Prediction.shape[0]):
    cv2.imwrite('T1Prediction/'+str(i)+'.png', Prediction[i,:,:,0])
    cv2.imwrite('GroundTruth/'+str(i)+'.png', GroundTruth[i,:,:])
    cv2.imwrite('PreContrast/'+str(i)+'.png', T1[i,:,:])


print ('Done')
