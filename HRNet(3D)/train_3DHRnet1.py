from keras import backend as K
import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import multi_gpu_model
from keras.callbacks import ReduceLROnPlateau
import h5py as h5
import keras
from keras.layers import Input
from keras.models import load_model
from hrnet_3D1 import *
from utils import *
from DataAugment import *
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3,4,5,6"
os.environ['OMP_NUM_THREADS']='10'
batch_size = 3
epochs = 6

'''load data'''
MRIData = h5.File('/raid/chenchao/code/3DHRNet/Dataset/MRIData_subset_3D_55_175.h5','r')
x_train = MRIData['x_train']
y_train = MRIData['y_train']
x_test = MRIData['x_test']
y_test = MRIData['y_test']
x_train = np.array(x_train)
y_train = np.expand_dims(y_train,axis=-1)
x_test = np.array(x_test)
y_test = np.expand_dims(y_test,axis=-1)

start = 50
end = 53
print ('x_train shape:', x_train.shape)
print ('y_train shape:', y_train.shape)

x_test = x_test[:,start:end,:,:,:]
y_test = y_test[:,start:end,:,:,0]
y_test = np.expand_dims(y_test,axis=-1)
print ('x_test shape:', x_test.shape)
print ('y_test shape:', y_test.shape)

deltaT1 = y_train[:,:,:,:,0] - x_train[:,:,:,:,0]
deltaT1 = np.expand_dims(deltaT1,axis=-1)
deltaT1[deltaT1>0.6] = 0
deltaT1[deltaT1>0.10] = 1
deltaT1[deltaT1<=0.1] = 0
for i in range(75):
    deltaT1[:,i,:,:,:] = 0
for i in range(deltaT1.shape[0]):
    for j in range(deltaT1.shape[1]):
        if np.sum(deltaT1[i,j,:,:,:]) > 10000:
            deltaT1[i,j,:,:,:] = 0

for i in range(368):
    print (i)
    cv2.imwrite('deltaT1/'+str(i)+'.png',255*deltaT1[i,90,:,:])
print ('deltaT1 shape:', deltaT1.shape)



def Generator(x_train,y_train,deltaT1,batch_size):
    m = 0
    n = 0
    depth = 3
    while True:
        m = np.random.choice(x_train.shape[0],batch_size)
        n = np.random.randint(0,x_train.shape[1]-depth)
        data_x = x_train[m, n:n+depth,:,:,:]
        data_y = y_train[m, n:n+depth,:,:,:]
        mask = deltaT1[m, n:n+depth,:,:,:]
        yield ([data_x,mask],data_y)


########################### Model Construction ######################
mask,model = HRnet(batch_size = batch_size, depth=3, height = 256, width = 192, channel = 3, classes =1)
test_mask = np.ones((42,3,256,192,1))

model.summary()
Model = multi_gpu_model(model,gpus=6)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='auto', factor=0.2, patience=3, min_lr=0.00000001)
Adam = keras.optimizers.Adam(lr=0.0001,beta_1=0.9,beta_2=0.999)
Model.compile(optimizer=Adam, loss=custom_loss1(mask), metrics=['MAE'])
# load the saved weights
#model.load_weights('model_3D_weights.h5')

#for num, layer in enumerate(model.layers[:-50]):
#    layer.trainable = False
#    print (num,layer.trainable)
history = Model.fit_generator(Generator(x_train,y_train,deltaT1,batch_size=18),steps_per_epoch=2000, epochs=epochs, verbose=1,validation_data=([x_test,test_mask],y_test))
#history = Model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test,y_test), callbacks=[reduce_lr])
score = Model.evaluate([x_test,test_mask],y_test,batch_size=batch_size)
print ('Test Loss:', score[0])
print ('Test MSE:', score[1])



model.save('model_3D.h5')
model.save_weights('model_3D_weights.h5')
print ('Done')
