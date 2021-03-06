# Contrast-enhanced MRI Synthesis Using 3D High-Resolution ConvNets

* This repository contains code for our paper **Contrast-enhanced MRI Synthesis Using 3D High-Resolution ConvNets** 

## Data
<div align=center><img src="https://github.com/chenchao666/Contrast-enhanced-MRI-Synthesis/blob/master/img/fig1.png" width="950" /></div>
Three non-contrast brain MRI scans, including T1, T2, and Apparent Diffusion Coefficient (ADC), are utilized as inputs. The contrast-enhanced T1 (CE-T1) is utilized as the ground truth image. We aim to synthesize the CE-T1 from the precontrast (zero-dose) MRI scans by training a 3D FCN generator.


## Model 
<div align=center><img src="https://github.com/chenchao666/Contrast-enhanced-MRI-Synthesis/blob/master/img/fig2.png" width="950" /></div>
The overview of our proposal. A high resolution FCN model was trained as a generator to synthesize the contrast-enhanced T1. Three non-contrast brain MRI scans, including T1, T2, and ADC, were utilized as input images.

## Results
<div align=center><img src="https://github.com/chenchao666/Contrast-enhanced-MRI-Synthesis/blob/master/img/fig3.png" width="950" /></div>
Qualitative evaluation of our proposal and other widely used baseline models. Only one typical slice is demonstrated for each subject. From left to right, input T1, input T2, input ADC, ground truth image CE-T1, synthetic image of the 2D U-Net,  synthetic image of the 3D U-Net, synthetic image of the proposed 2D FCN model, synthetic image of the proposed 3D FCN model, the absolute difference between the results of our 3D FCN and the ground truth. The first two rows are from normal patients and the other rows are from patients with tumors. Different rows are from different subjects in the test set.


<div align=center><img src="https://github.com/chenchao666/Contrast-enhanced-MRI-Synthesis/blob/master/img/fig4.png" width="950" /></div>
Visual assessment of our proposal in two representative test samples in set B. From top to bottom, the ground truth slices of test patient (a), the synthetic CE-T1 for test patient (a), the ground truth slices of test patient (b), the synthetic CE-T1 for test patient (b). The synthetic CE-T1 images are generated by the proposed 3D FCN model. The images in the same row represent different slices of the same subjects.

