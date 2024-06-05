# Project Proposal - MLinAPP

## Project Title, Project Number and Project Owner
AM04, Dataset Augmentation for Additive Manufacturing defect detection - 2024/02  - Alessio Mascolini

## Project Team
Gabriele Ferro (s308552), Giovanni Gaddi (s308685), Filippo Greco (s309529), Stiven Hidri (s315147)

## Project Background
The project came by the need for bigger datasets when data is scarce. Our team, composed of the above-mentioned 4 people, will work on the described project, after it was proposed by Alessio Masolini - the project owner. One of the main challenges will be applying augmentation techniques that we have never seen before. We were motivated especially by the industrial application of the project and by the urgency of companies to augment datasets when their base is not large enough. 

## Project Scope
Ideally, we want to pursuit the following goals:
1.	Segment manually the images with defects, to identify precisely where the defects are in the images. 
2.	Generate composite images by applying color transferring methods to the defects (foreground) without changing the background. These images, along with the original ones, will be used as training data for an end-to-end deep convolutional neural network based on encoder-decoder able to harmonize the artificial generated images based on Tsai Deep Image Harmonization. Unlike the referred model we won’t be implement the Scene Parsing Decoder because this is used only to give context (through labels) on the image when different domains are taken into consideration.
3.	We will insert random defects into no-defect images and use the model described at point 2 to harmonize these crafted images. This process makes the background and the injected defects compatible in terms of light, color, and other visual attributes and such resulting in realistic examples of defect images.
4.	Utilizing an already existing CNN-based image classifier, we would like to compare the performance using only the original dataset and the augmented one with the previously mentioned techniques, with the latter that should perform better.


## High-Level Requirements
HW Requirements:
We kindly request the use of PoliTO servers for computational resources to expedite the testing of our models. While it is possible to run the workload on some of our machines, we cannot be certain that they will be sufficient. Additionally, some of our members do not have GPUs on their PCs.
SW Requirements:
1.	VS Code text editor with the SSH plugin for remote coding on PoliTO servers.
2.	Public GitHub repository available at: https://github.com/figimodi/AML-egovision
3.	Telegram for internal communications among members.

## Implementation Plan
We would like to implement the following solutions:
1. Classification Training: we will train a ResNet50 Model (and/or eventually some similar ones in case of poor results with ResNet50) on the existing no-augmented dataset, to evaluate the classification model before the augmentation. This evaluation will be used as a comparison afterword.
2. Segmentation: given the restricted number of images we will manually make masks highlighting the defects.
3. Generation: apply the linear color transferring methods (modify contrast, light, saturation, …) to the defects from the relative mask. 
4. Image Harmonization Model Training: we will train the Autoencoder based on the Tsai Deep Image Harmonization Model, experimenting with various color transferring techniques.
5. Dataset Augmentation: we will randomly apply random existing defects on no-defects images, which will then be harmonized with the previously trained model. 
6. Classification Training: we will train a ResNet50 Model on the new augmented dataset.
7. Evaluation: By testing the classification models we will compare the results obtained with both datasets, augmented and not, supposedly noticing an increase in performance using the augmented one.


## Tentative Timeline/Schedule
Activity Description | Duration
--- | --- 
Manual segmentation and implementation linear color transfer | 2 week
Implementation of the classification networks | 1 week
Implementation of the composing algorithm | 1 week
Implementation + experimentation of harmonization algorithms | 3 weeks
Training the networks and comparing results | 3 weeks
Report writing | 3 weeks
