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
import nibabel as nib
import os
os.environ["CUDA_VISIBLE_DEVICES"]="4"
os.environ['OMP_NUM_THREADS']='10'
batch_size = 1

'''load data'''
# TestData: [T1,T2,ADC];  TestMask: T1c;  Mask: Brain Mask
TestData = np.load('/raid/chenchao/code/3DHRNet/Dataset/TestData.npy')
TestMask = np.load('/raid/chenchao/code/3DHRNet/Dataset/TestPredict.npy')
Mask = np.load('/raid/chenchao/code/3DHRNet/Dataset/TestMask.npy')

TestData = np.swapaxes(TestData,1,3)
TestMask = np.swapaxes(TestMask,1,3)
Mask = np.swapaxes(Mask,1,3)

print (TestData.shape)
print (TestMask.shape)
start = 70
end = 175
num = int((end-start)/3)


Model = load_model('model_3D.h5',custom_objects={'custom_loss':custom_loss})

def my_ssim(GroundTruth,Prediction):
    SSIM = []
    for i in range(GroundTruth.shape[0]):
        ans = ssim(GroundTruth[i,:,:],Prediction[i,:,:],data_range=1.0)
        SSIM.append(ans)
    return np.mean(np.array(SSIM))



def Test(Model,TestData,TestMask,Mask):
    TestImg = TestData[start:end,:,:,:]
    TestImg = np.expand_dims(TestImg,axis=0)
    GroundTruth = TestMask[start:end,:,:,0]
    GroundTruth = np.expand_dims(GroundTruth,axis=0)
    GroundTruth = np.expand_dims(GroundTruth,axis=4)
    Mask = Mask[start:end,:,:]
    Mask = np.expand_dims(Mask,axis=0)
    Mask = np.expand_dims(Mask,axis=4)
    Prediction = np.zeros(GroundTruth.shape)
    for i in range(num):
        Img = TestImg[0,i*3:(i+1)*3,:,:,:]
        Img = np.expand_dims(Img,axis=0)
        Prediction[0,i*3:(i+1)*3:,:,:] = Model.predict(Img)
    print (Prediction.shape)
    print (Mask.shape)
    MAE = np.abs(Prediction-GroundTruth)*Mask
    MAE = np.mean(MAE)*np.size(Mask)/np.sum(Mask)
    MSE = np.square(Prediction-GroundTruth)*Mask
    MSE = np.mean(MSE)*np.size(Mask)/np.sum(Mask)
    PSNR = 10*np.log10(1.0/MSE)
    SSIM = my_ssim(GroundTruth[0,:,:,:,0],Prediction[0,:,:,:,0])
    return MAE,PSNR,SSIM

# index denotes the index of patients with tumors
index = [1,5,11,12,15,16,17,18,21,22,24,37,39]
for ind in index:
    TestImg = TestData[ind,start:end,:,:,:]
    TestImg = np.expand_dims(TestImg,axis=0)
    GroundTruth = TestMask[ind,start:end,:,:,0]
    print (np.max(GroundTruth))
    print (np.min(GroundTruth))
    GroundTruth = np.expand_dims(GroundTruth,axis=3)
    Input = TestData[ind,start:end,:,:,:]
    Prediction = np.zeros(GroundTruth.shape)
    for i in range(num):
        print (i)
        Img = TestImg[0,i*3:(i+1)*3,:,:,:]
        Img = np.expand_dims(Img,axis=0)
        Output = Model.predict(Img)
        print (Output.shape)
        Prediction[i*3:(i+1)*3,:,:,:] = Output
    diff = np.abs(Prediction - GroundTruth)
    #MAE = np.mean(diff,axis=(1,2,3))
    GroundTruth = 255*GroundTruth
    Input = 255*Input
    print (Prediction.shape)
    print (GroundTruth.shape)
    Prediction = 255*Prediction
    for i in range(Prediction.shape[0]):
        print (i)
        cv2.imwrite('T1Prediction/'+str(i)+'.png', Prediction[i,:,:,0])
        cv2.imwrite('GroundTruth/'+str(i)+'.png', GroundTruth[i,:,:,0])
        cv2.imwrite('Subtraction/'+str(i)+'.png',255*diff[i,:,:,0])
        cv2.imwrite('PreContrast/'+str(i)+'0.png', Input[i,:,:,0])
        cv2.imwrite('PreContrast/'+str(i)+'1.png',Input[i,:,:,1])
        cv2.imwrite('PreContrast/'+str(i)+'2.png',Input[i,:,:,2])
    Prediction_New = np.swapaxes(Prediction,0,2)
    GroundTruth_New = np.swapaxes(GroundTruth,0,2)
    NiftiImg = nib.Nifti1Image(Prediction_New[:,:,:,0], affine=np.eye(4))
    nib.save(NiftiImg,'nifiti/'+str(ind)+'-T1cPrediction'+'.nii.gz')
    NiftiImg = nib.Nifti1Image(GroundTruth_New[:,:,:,0], affine=np.eye(4))
    nib.save(NiftiImg,'nifiti/'+str(ind)+'-T1cGroundTruth'+'.nii.gz')


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


#NiftiImg = nib.Nifti1Image(Prediction[:,:,:,0], affine=np.eye(4))
#nib.save(NiftiImg,'nifiti/'+str(ind)+'-T1cPrediction'+'.nii.gz')
#NiftiImg = nib.Nifti1Image(GroundTruth[:,:,:,0], affine=np.eye(4))
#nib.save(NiftiImg,'nifiti/'+str(ind)+'-T1cGroundTruth'+'.nii.gz')

