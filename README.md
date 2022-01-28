# Content loss and conditional space relationship in conditional generative adversarial networks

This repository is associated with the paper titled as "Content loss and conditional space relationship in conditional generative adversarial networks" submitted to 
TURKISH JOURNAL OF ELECTRICAL ENGINEERING & COMPUTER SCIENCES  and  currently under the review stage. 

In this paper, we investigated the behaviour of GAN in the case of shrinking the conditional space and increasing the computational capacity of the generator network. 

As a baseline, we used Pix2pix GAN model, which is explained at Scenario 1. 
![pix2pixDiagram](https://user-images.githubusercontent.com/22565098/151542356-6c00c0a9-26e6-4ca8-a08f-f85b9db07c54.png)




At scenario 2, we used two sequential generators and one discriminator as depicted at the figure below;
![scenario2Diagram](https://user-images.githubusercontent.com/22565098/151543406-14d8dc26-a7cc-407a-a7a6-5b5d15cc2df8.png)
With this design, we aim to shrink the conditional space of the second generator and investigated the effect of this on the content loss. 


At scenario 3, we simply doubled the size of the generator used in Scenario-1 and monitored the effect of this on the content loss.

At the final scenario 4, we used two sequential generators with two discriminators and examined the effect of this design choice on the contet loss.
![scenario4Diagram](https://user-images.githubusercontent.com/22565098/151543930-dcd50657-f117-4729-89b6-f437bcdbb942.png)


# Acknowledgments
Code borrows heavily from DCGAN and Pix2Pix GAN models.
original implementations can be found at 
https://github.com/phillipi/pix2pix and https://machinelearningmastery.com
