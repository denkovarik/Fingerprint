# Fingerprint

![Sample Images](https://github.com/denkovarik/Fingerprint/blob/main/images/GAN%20Generated%20Images.PNG)

## Introduction
An open problem in the area of Latent Fingerprint Recognition is the enhancement of poor quality fingerprints. Although there are alot of algorithms out there to enhance fingerprints, the results for some of the best solutions are less than satisfactory when used on latent fingerprints. For this reason, my research was on the use of Generative Adversarial Networks for the purpose of enhacing images of latent fingerprints for improving matching accuracy.

The GAN in this repo was adapted from [PyTorch DCGAN TUTORIAL](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html). It is similar to traditional GANs, but its discriminator consists of 2 parts. The first is the normal discriminator that attempts to determine if a generated image is real of fake. The second part of this GAN is a Siamese Neural Network, which is used to try and enforce that there is similair ridge structure between the original input fingerprint image and the GAN Enhanced fingerprint. This Siamese Neural Network was adapted from [one-shot-siamese](https://github.com/kevinzakka/one-shot-siamese) for comparing images or characters.

The loss from these 2 parts of the discriminator are combined to determine the total loss for the discriminator. In doing this, the hope is that the trained GAN would be able to Generate realistic binary enhanced fingerprints from poor fingerprint images, while also preserving the underlying ridge structure.

## The Dataset
It was difficult to get access to large enough fingerprint datasets for training. This is because fingerprints are considered personal information, so this data is not commonly avaiable to everyone. Because of this, I ended up synthetically generating my own dataset using [Anguli](https://dsl.cds.iisc.ac.in/projects/Anguli/). This generated dataset contains close to one million fingerprint images of varying qualities, which includes 10,000 unique fingerprints. You should be able to download the dataset using the following link: [Prepped_Fingerprints_206x300.zip](https://drive.google.com/file/d/1DZVQVEDQeghQp61zOUuovzkZCyFcmevX/view?usp=share_link). This dataset is over 17 GB in size, so be warned.

Below are some sample images that the GAN will be trained on. The top row represents the input fingerprint images that the generator is to enhanced. The bottom row are the enhanced versions of the fingerprints above, where were enhanced using Gabor Filters (a common method for enhancing fignerprint images).

![Sample Images](https://github.com/denkovarik/Fingerprint/blob/main/images/datasetSample.PNG)

## Usage
This project is currently being developed in Paperspace. You can access this Notebook using the following link: [Fingerprint](https://console.paperspace.com/denkovarik123/notebook/r8krvughxoashik).

## Sample Results 
Below are some sample images that were generated by the GAN in this project. This was just a normal GAN that doesn't use a Siamese Nueral Network for the discriminator. The top row is the input images that were fed into the GAN. The middle row shows the enhanced images that the GAN produce, and the bottom row represents the ideal enhanced versions of the images in the top row. Please note that this bottom row just represents the ideal enhanced versions of the fingerprints from the top row. The images in this bottom row were produced by using Gabor Filters to enhance the good impressions of the fingerprints (and not necessarly the impressions shown in the top row).

![Sample Results](https://github.com/denkovarik/Fingerprint/blob/main/images/generated%20fingerprints%20comp%2020230220.PNG)

## Notebooks

### [Fingerprint Siamese Neural Network 2.0 for Unenhanced Fingerprint Images](https://denkovarik.github.io/Fingerprint/Experiments/Fingerprint%20Siamese%20Neural%20Network%202%2020230308.html) (3/8/2023)

Siamese Neural Network that compares the similarity between two unenhanced fingerpints. Better performance was achieved by increasing the number of channels per layer. It was able to achieve a Testing Accuracy of 97%, and it was able to consistenly achieve a validateion accuracy of over 96% for the last 80 epochs of training. Although the model had to be trained for much longer than the previous version, the increased size of the network seemed to help avoid overfitting since the validation accuracy was consistenly similar to that of the training accuracy. In addition, the model achieved a testing accuracy that was higher than the training accuracy from the previous epoch, so it was able to generalize the information it learned from training.

One limitation of this model is that it may have been able to exploit some artifact found in the synthetically generated fingerprints. Only testing this model on a dataset of real fingerprints would tell for sure though.

### [Fingerprint Siamese Neural Network for Unenhanced Fingerprint Images](https://denkovarik.github.io/Fingerprint/Experiments/Fingerprint%20Siamese%20Neural%20Network%2020230227.html) (2/27/2023)

Training for Siamese Neural Network on unenhanced fingeprint images of size 300 x 206.  

The performance of the Fingerprint Siamese Neural Network turned out better than expected. It achieved a test accuracy of 93%, which is really suprising considering that the dataset contains some really bad fingerprint impressions. Assuming that the model is not exploiting some artifact found in the synthetically generated fingerprints, it appears that the subnetwork is able to reliably extract the features from these unenhanced fingerprint impressions. Maybe this trained subnetwork could be used in training the GAN? It could be possible to use the trained Fingerprint Siamese Neural Network subnetwork to produce a latent vector of fingerprint features that the GAN would then use to generate images of fingerprints from. This would allow for the generator to have a much shallower network, which would simplify it's network architecture and possibly help avoid mode collapse.

### [Fingerprint GAN with Class Grouping](https://denkovarik.github.io/Fingerprint/Experiments/Fingerprint%20GAN_20230220.html) (2/20/2023)

Training for a normal GAN (no Siamese Neural Network Used) GAN on images of size 300 x 300. Previous iterations of this GAN had problems with experience mode collapse. The model would appear to start producing realistic images of fingerprints, but then mode collapse will occur and nothing but a grey image gets produced. An attempt was made to address this issue by grouping the classes. This was done by modifying the dataloader to only return one unique fingerprint per batch. In addition, the model architeture was modified, and the learing rate for the discrimator was decreased by 10. 

It appears that better results were achieved by grouping the classes, modifying the network architecture, and reducing the learning rate of the discriminator. After 50 epochs of training this GAN, convincing fingerprint images are being produced, and mode collapse has so far been avoided. Unfortunately the training loss still seems to increase and fluctuate more as time goes on. This makes it seem like mode collapse will still eventually occur after enough training has elapsed. Only more training will tell for sure.
 
### [Fingerprint Siamese Neural Network for Enhanced Fingerprint Images](https://denkovarik.github.io/Fingerprint/Experiments/Fingerprint%20Siamese%20Neural%20Network_20230215.html) (2/15/2023)

Training for Siamese Neural Network on images of size 300 x 300.  

It appears that the model was able to achieve decent performance. The model acheived a validation accuracy of 98% during training, and it correctly classified each pair of fingerprints above as the same or different (at least in most cases). One limitation of this model though could be the data that it was trained on. This project assumes that the fingerprints in the dataset are representative of real fingerprints (at least for good impression fingerprints). If there is some problem with the software that was used to generate the fingerprints in this dataset, then the model may not preform will when used on enhanced images of real (good quality) fingerprints.

### [Fingerprint GAN 1.0](https://denkovarik.github.io/Fingerprint/Experiments/Fingerprint%20GAN_20230212.html) (2/12/2023)

Training for a normal GAN (no Siamese Neural Network Used) GAN on images of size 256 x 256.   

Initial result in training looked promising, but mode collapse prevented the GAN from being trained until the desired performance was reached. This happened at around epoch 6, and it seemed to have been caused by the disciminator outcompeting the generator. It is possible that the discriminator is powerful enough to be trained to distinquish different classes from the dataset.  

One way to fix this could be to group the classes in the dataset. This could be done by modifying the dataloader to only provide mulitiple impressions of the same fingerprint (instead of impressions of multiple different fingerprints) in each batch. This would allow the discriminator to classify each batch as real or fake. 

## Author
Dennis Kovarik
