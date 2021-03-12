import numpy as np
import cv2


def Mixup(x_tumor,y_tumor,num=30000):
    alpha =0.2
    X_mixup = []
    Y_mixup = []
    lam = np.random.beta(alpha,alpha,num)
    for i in range(num):
        print (i)
        ind = np.random.choice(x_tumor.shape[0],2,replace=False)
        print (ind)
        x1 = x_tumor[ind[0],:,:,:]
        x2 = x_tumor[ind[1],:,:,:]
        y1 = y_tumor[ind[0],:,:,:]
        y2 = y_tumor[ind[1],:,:,:]
        X_mixup.append(lam[i]*x1 + (1-lam[i])*x2)
        Y_mixup.append(lam[i]*y1 + (1-lam[i])*y2)
    X_mixup = np.array(X_mixup)
    Y_mixup = np.array(Y_mixup)
    print (X_mixup.shape)
    return X_mixup, Y_mixup



def CrossMixup(x_train,y_train,x_tumor,y_tumor,num=80000):
    num = num
    alpha =0.2
    X_mixup = np.zeros((num,x_tumor.shape[1],x_tumor.shape[2],x_tumor.shape[3]))
    Y_mixup = np.zeros((num,y_tumor.shape[1],y_tumor.shape[2],y_tumor.shape[3]))
    lam = np.random.beta(alpha,alpha,num)
    for i in range(num):
        print (i)
        ind1 = np.random.choice(x_tumor.shape[0],1,replace=False)
        ind2 = np.random.choice(x_train.shape[0],1,replace=False)
        x1 = x_tumor[ind1,:,:,:]
        x2 = x_train[ind2,:,:,:]
        y1 = y_tumor[ind1,:,:,:]
        y2 = y_train[ind2,:,:,:]
        X_mixup[i,:,:,:] = lam[i]*x1 + (1-lam[i])*x2
        Y_mixup[i,:,:,:] = lam[i]*y1 + (1-lam[i])*y2
    return X_mixup, Y_mixup






if __name__ == '__main__':
    x_tumor = np.load('tumorData1.npy')
    y_tumor = np.load('tumorMask1.npy')
    x_tumor,y_tumor=Mixup(x_tumor,y_tumor)
    print (np.max(y_tumor))
    print (y_tumor.shape)
    np.save('tumorData1_mixup.npy',x_tumor)
    np.save('tumorMask1_mixup.npy',y_tumor)
