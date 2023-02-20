# Fingerprint

This repo contains some of the work from my unfinished Master's degree from 2021. I ended up lossig some of my work, so I had to create it. 

## Introduction
An open problem in the area of Latent Fingerprint Recognition is the enhancement of poor quality fingerprints. Although there are alot of algorithms out there to enhance fingerprints, the results for some of the best solutions are less than satisfactory when used on latent fingerprints. For this reason, my research was on the use of Generative Adversarial Networks for the purpose of enhacing images of latent fingerprints for improving matching accuracy.

The GAN in this repo was adapted from [PyTorch DCGAN TUTORIAL](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html). It is similar to traditional GANs, but its discriminator consists of 2 parts. The first is the normal discriminator that attempts to determine if a generated image is real of fake. The second part of this GAN is a Siamese Neural Network, which is used to try and enforce that there is similair ridge structure between the original input fingerprint image and the GAN Enhanced fingerprint. This Siamese Neural Network was adapted from [one-shot-siamese](https://github.com/kevinzakka/one-shot-siamese) for comparing images or characters.

The loss from these 2 parts of the discriminator are combined to determine the total loss for the discriminator. In doing this, the hope is that the trained GAN would be able to Generate realistic binary enhanced fingerprints from poor fingerprint images, while also preserving the underlying ridge structure.

## The Dataset
It was difficult to get access to large enough fingerprint datasets for training. This is because fingerprints are considered personal information, so this data is not commonly avaiable to everyone. Because of this, I ended up synthetically generating my own dataset using [Anguli](https://dsl.cds.iisc.ac.in/projects/Anguli/). This generated dataset contains close to one million fingerprint images of varying qualities, which includes 10,000 unique fingerprints. 

Below are some sample images that the GAN will be trained on. The top row represents the input fingerprint images that the generator is to enhanced. The bottom row are the enhanced versions of the fingerprints above, where were enhanced using Gabor Filters (a common method for enhancing fignerprint images).

![Sample Images](https://github.com/denkovarik/Fingerprint/blob/main/images/datasetSample.PNG)

## Usage
This project is currently being developed in Paperspace. You can access this Notebook using the following link: [Fingerprint](https://console.paperspace.com/denkovarik123/notebook/r8krvughxoashik).

## Results (so far)
Below are some sample images that were generated by the GAN in this project. The top row is the input images that were fed into the GAN. The middle row shows the enhanced images that the GAN produce, and the bottom row represents the ideal enhanced versions of the images in the top row. Please note that this bottom row just represents the ideal enhanced versions of the fingerprints from the top row. The images in this bottom row were produced by using Gabor Filters to enhance the good impressions of the fingerprints (and not necessarly the impressions shown in the top row).

![Sample Results](https://github.com/denkovarik/Fingerprint/blob/Dev/images/generated%20fingerprints.PNG)

## Experiments
### <ins>2/12/2023</ins>  
Training for a normal GAN (no Siamese Neural Network Used) GAN on images of size 256 x 256.    

[Fingerprint GAN 2/12/2023](https://denkovarik.github.io/Fingerprint/Experiments/Fingerprint%20GAN_20230212.html)  

Initial result in training looked promising, but mode collapse prevented the GAN from being trained until the desired performance was reached. This happened at around epoch 6, and it seemed to have been caused by the disciminator outcompeting the generator. It is possible that the discriminator is powerful enough to be trained to distinquish different classes from the dataset.  

One way to fix this could be to group the classes in the dataset. This could be done by modifying the dataloader to only provide mulitiple impressions of the same fingerprint (instead of impressions of multiple different fingerprints) in each batch. This would allow the discriminator to classify each batch as real or fake. 

### <ins>2/15/2023</ins>  
Training for Siamese Neural Network on images of size 300 x 300.    

[Fingerprint Siamese NN 2/12/2023](https://denkovarik.github.io/Fingerprint/Experiments/Fingerprint%20Siamese%20Neural%20Network_20230215.html)  

It appears that the model was able to achieve decent performance. The model acheived a validation accuracy of 98% during training, and it correctly classified each pair of fingerprints above as the same or different (at least in most cases). One limitation of this model though could be the data that it was trained on. This project assumes that the fingerprints in the dataset are representative of real fingerprints (at least for good impression fingerprints). If there is some problem with the software that was used to generate the fingerprints in this dataset, then the model may not preform will when used on enhanced images of real (good quality) fingerprints.

## Author
Dennis Kovarik
