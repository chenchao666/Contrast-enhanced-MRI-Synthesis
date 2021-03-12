import numpy as np
from keras.models import load_model
from keras import backend as K
import pickle
import numpy as np
import matplotlib.pyplot as plt
import keras
import h5py as h5
from utils import *
from skimage.metrics import structural_similarity as ssim
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['OMP_NUM_THREADS']='10'
batch_size = 1

'''load data'''
#TrainData = np.load('mask/TrainData.npy')
#TrainMask = np.load('mask/TrainPredict.npy')
TestData = np.load('/raid/chenchao/code/brainMRI/mask/TestData.npy')
TestMask = np.load('/raid/chenchao/code/brainMRI/mask/TestPredict.npy')
Mask = np.load('/raid/chenchao/code/brainMRI/mask/TestMask.npy')

TestData = np.swapaxes(TestData,1,3)
TestMask = np.swapaxes(TestMask,1,3)
Mask = np.swapaxes(Mask,1,3)

print (TestData.shape)
print (TestMask.shape)
start = 60
end = 140

Model = load_model('model_3D.h5',custom_objects={'custom_loss':custom_loss})

def my_ssim(GroundTruth,Prediction):
    SSIM = []
    for i in range(GroundTruth.shape[0]):
        ans = ssim(GroundTruth[i,:,:],Prediction[i,:,:],data_range=1.0)
        SSIM.append(ans)
    return np.mean(np.array(SSIM))



def Test(Model,TestData,TestMask,Mask):
    start = 60
    end = 140
    TestImg = TestData[start:end,:,:,:]
    TestImg = np.expand_dims(TestImg,axis=0)
    GroundTruth = TestMask[start:end,:,:,0]
    GroundTruth = np.expand_dims(GroundTruth,axis=0)
    GroundTruth = np.expand_dims(GroundTruth,axis=4)
    Mask = Mask[start:end,:,:]
    Mask = np.expand_dims(Mask,axis=0)
    Mask = np.expand_dims(Mask,axis=4)
    Prediction = Model.predict(TestImg)
    print (Prediction.shape)
    print (Mask.shape)
    MAE = np.abs(Prediction-GroundTruth)*Mask
    MAE = np.mean(MAE)*np.size(Mask)/np.sum(Mask)
    MSE = np.square(Prediction-GroundTruth)*Mask
    MSE = np.mean(MSE)*np.size(Mask)/np.sum(Mask)
    PSNR = 10*np.log10(1.0/MSE)
    SSIM = my_ssim(GroundTruth[0,:,:,:,0],Prediction[0,:,:,:,0])
    return MAE,PSNR,SSIM


ind = 24
TestImg = TestData[ind,start:end,:,:,:]
TestImg = np.expand_dims(TestImg,axis=0)
GroundTruth = TestMask[ind,start:end,:,:,0]
print (np.max(GroundTruth))
print (np.min(GroundTruth))
GroundTruth = np.expand_dims(GroundTruth,axis=3)
Input = TestData[ind,start:end,:,:,:]
Prediction = Model.predict(TestImg)
diff = np.abs(Prediction - GroundTruth)
#MAE = np.mean(diff,axis=(1,2,3))
GroundTruth = 255*GroundTruth
Input = 255*Input
print (Prediction.shape)
print (GroundTruth.shape)
Prediction = 255*Prediction
for i in range(Prediction.shape[1]):
    print (i)
    cv2.imwrite('T1Prediction/'+str(i)+'.png', Prediction[0,i,:,:,0])
    cv2.imwrite('GroundTruth/'+str(i)+'.png', GroundTruth[i,:,:,0])
    cv2.imwrite('substraction/'+str(i)+'.png',255*diff[0,i,:,:,0])
    cv2.imwrite('PreContrast/'+str(i)+'0.png', Input[i,:,:,0])
    cv2.imwrite('PreContrast/'+str(i)+'1.png',Input[i,:,:,1])
    cv2.imwrite('PreContrast/'+str(i)+'2.png',Input[i,:,:,2])

MAE = []
PSNR = []
SSIM = []
for i in range(TestData.shape[0]):
    mae,psnr,_ssim = Test(Model,TestData[i,:,:,:,:],TestMask[i,:,:,:,:],Mask[i,:,:,:])
    MAE.append(mae)
    PSNR.append(psnr)
    SSIM.append(_ssim)
print ('MAE:', np.mean(np.array(MAE)))
print ('PSNR:', np.mean(np.array(PSNR)))
print ('SSIM:', np.mean(np.array(SSIM)))



