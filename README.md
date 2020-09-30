# Single-Image-De-Raining-Keras-
Implemented Image De-raining Using a  Conditional Generative Adversarial Network using Keras

[[Paper Link](https://arxiv.org/abs/1701.05957)]

Severe weather conditions such as rain and snow adversely affect the visual quality of images captured under such conditions thus rendering them useless for further usage and sharing. In addition, such degraded images drastically affect performance of vision systems. Hence, it is important to solve the problem of single image de-raining/de-snowing. 

In this project I  implemented the Paper using  Keras.

It is kept in mind that the de-rained result should be indistinguishable from its corresponding clear image . 

	@article{zhang2017image,		
	  title={Image De-raining Using a Conditional Generative Adversarial Network},
	  author={Zhang, He and Sindagi, Vishwanath and Patel, Vishal M},
	  journal={arXiv preprint arXiv:1701.05957},
	  year={2017}
	} 
  
  <img src="image/example1.jpg" width="400px" height="200px"/><img src="image/example2.jpg" width="400px" height="200px"/>
  
  
  *Make sure to change scipy version to 1.1.0
  
  Download the dataset from (https://drive.google.com/open?id=0Bw2e6Q0nQQvGbi1xV1Yxd09rY2s) 
 
 ##Acknowledgments## 
 
 Code borrows heavily  from [[Pix2Pix Peceptual Loss](https://github.com/ssajj1212/Pix2Pix-keras)]
 and [[Pix2Pix](https://github.com/hanwen0529/GAN-pix2pix-Keras)]. Thanks for the sharing.
