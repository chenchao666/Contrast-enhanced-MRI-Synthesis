import numpy as np
from keras.models import load_model
from keras import backend as K
import pickle
import numpy as np
import matplotlib.pyplot as plt
import keras
import h5py as h5
from unet import *
from utils import *
import cv2
import os
from skimage.metrics import structural_similarity as ssim
os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ['OMP_NUM_THREADS']='10'
batch_size = 128

'''load data'''
#TrainData = np.load('mask/TrainData.npy')
#TrainMask = np.load('mask/TrainPredict.npy')
TestData = np.load('/raid/chenchao/code/brainMRI/mask/TestData.npy')
TestMask = np.load('/raid/chenchao/code/brainMRI/mask/TestPredict.npy')
Mask = np.load('/raid/chenchao/code/brainMRI/mask/TestMask.npy')

#Train = h5.File('/raid/jpolson/braintumor/3DT1Cprediction_Train.hdf5','r')
#Test = h5.File('/raid/jpolson/braintumor/3DT1Cprediction_Test.hdf5','r')
#TrainData = Train['data']

#TrainData = np.swapaxes(TrainData,1,3)
#TrainMask = Train['mask']
#TrainMask = np.swapaxes(TrainMask,1,3)
#TestData = Test['data']
TestData = np.swapaxes(TestData,1,3)
#TestMask = Test['mask']
TestMask = np.swapaxes(TestMask,1,3)
Mask = np.swapaxes(Mask,1,3)
#TrainData = 0.5*(TrainData+1)
#TrainMask = 0.5*(TrainMask+1)
#TestData = 0.5*(TestData+1)
#TestMask = 0.5*(TestMask+1)
#print (TrainData.shape)
#print (TrainMask.shape)
print (TestData.shape)
print (TestMask.shape)
start = 60
end = 140

Model = load_model('model_unet.h5',custom_objects={'custom_loss':custom_loss})

def my_ssim(GroundTruth,Prediction,Mask):
    SSIM = []
    for i in range(GroundTruth.shape[0]):
        ans = ssim(GroundTruth[i,:,:,0],Prediction[i,:,:,0],data_range=1.0)
        SSIM.append(ans)
    return np.array(SSIM)



def Test(Model,TestData,TestMask,Mask):
    start = 60
    end = 140
    TestImg = TestData[:,start:end,:,:,:]
    TestImg = np.reshape(TestImg,(TestImg.shape[0]*TestImg.shape[1],TestImg.shape[2],TestImg.shape[3],TestImg.shape[4]))
    GroundTruth = TestMask[:,start:end,:,:,0]
    GroundTruth = np.reshape(GroundTruth,(GroundTruth.shape[0]*GroundTruth.shape[1],GroundTruth.shape[2],GroundTruth.shape[3],1))
    Mask = Mask[:,start:end,:,:]
    Mask = np.reshape(Mask,(Mask.shape[0]*Mask.shape[1],Mask.shape[2],Mask.shape[3],1))
    Prediction = Model.predict(TestImg)
    print (Prediction.shape)
    MAE = np.abs(Prediction-GroundTruth)*Mask
    MSE =np.square(Prediction-GroundTruth)*Mask
    MAE_ = []
    MSE_ = []
    PSNR_ = []
    for i in range(MAE.shape[0]):
        MAE_i = np.mean(MAE[i,:,:,0])*np.size(Mask[i,:,:,0])/np.sum(Mask[i,:,:,0])
        MSE_i = np.mean(MSE[i,:,:,0])*np.size(Mask[i,:,:,0])/np.sum(Mask[i,:,:,0])
        PSNR_i = 10*np.log10(1.0/MSE_i)
        MAE_.append(MAE_i)
        MSE_.append(MSE_i)
        PSNR_.append(PSNR_i)
#    MAE_mean = np.mean(MAE)*np.size(Mask)/np.sum(Mask)
#    MAE_std = np.std(MAE)*np.size(Mask)/np.sum(Mask)
#    MSE = np.square(Prediction-GroundTruth)*Mask
#    MSE_mean = np.mean(MSE)*np.size(Mask)/np.sum(Mask)
#    PSNR_mean = 10*np.log10(1.0/MSE_mean)
    print (PSNR_)
    SSIM = my_ssim(GroundTruth,Prediction,Mask)
    print ('MAE:',np.mean(MAE_),np.std(MAE_))
    print ('MSE:',np.mean(MSE_))
    print ('PSNR:',np.mean(PSNR_),np.std(PSNR_))
    print ('SSIM:',np.mean(SSIM),np.std(SSIM))



frame =99
TestImg = TestData[:,frame,:,:,:]
GroundTruth = TestMask[:,frame,:,:,0]
print (np.max(GroundTruth))
print (np.min(GroundTruth))
GroundTruth = np.expand_dims(GroundTruth,axis=4)
Input = TestData[:,frame,:,:,:]
Prediction = Model.predict(TestImg)
diff = np.abs(Prediction - GroundTruth)
MAE = np.mean(diff,axis=(1,2,3))
print (MAE)
GroundTruth = 255*GroundTruth
Input = 255*Input
print (Prediction.shape)
print (GroundTruth.shape)
Prediction = 255*Prediction
for i in range(Prediction.shape[0]):
    print (i)
    print (MAE[i])
    cv2.imwrite('T1Prediction/'+str(i)+'.png', Prediction[i,:,:,0])
    cv2.imwrite('GroundTruth/'+str(i)+'.png', GroundTruth[i,:,:,0])
    cv2.imwrite('substraction/'+str(i)+'.png',0.5*255*diff[i,:,:,0])
    cv2.imwrite('PreContrast/'+str(i)+'0.png', Input[i,:,:,0])
    cv2.imwrite('PreContrast/'+str(i)+'1.png',Input[i,:,:,1])
    cv2.imwrite('PreContrast/'+str(i)+'2.png',Input[i,:,:,2])

Test(Model,TestData,TestMask,Mask)
