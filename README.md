# Contrast-enhanced MRI Synthesis Using 3D High-Resolution ConvNets

## This repository contains code for our paper **Contrast-enhanced MRI Synthesis Using 3D High-Resolution ConvNets** [Download paper here](https://arxiv.org/abs/1912.11976)

## Data
<div align=center><img src="https://github.com/chenchao666/Contrast-enhanced-MRI-Synthesis/blob/master/img/fig1.png" width="950" /></div>
Three non-contrast brain MRI scans, including T1, T2, and Apparent Diffusion Coefficient (ADC), are utilized as inputs. The contrast-enhanced T1 (CE-T1) is utilized as the ground truth image. We aim to synthesize the CE-T1 from the precontrast (zero-dose) MRI scans by training a 3D FCN generator.


## Model 
<div align=center><img src="https://github.com/chenchao666/Contrast-enhanced-MRI-Synthesis/blob/master/img/fig2.png" width="950" /></div>
The overview of our proposal. A high resolution FCN model was trained as a generator to synthesize the contrast-enhanced T1. Three non-contrast brain MRI scans, including T1, T2, and ADC, were utilized as input images.

## Results
<div align=center><img src="https://github.com/chenchao666/Contrast-enhanced-MRI-Synthesis/blob/master/img/fig3.png" width="950" /></div>


<div align=center><img src="https://github.com/chenchao666/Contrast-enhanced-MRI-Synthesis/blob/master/img/fig4.png" width="950" /></div>

