from __future__ import print_function, division
import scipy

#Import Require Libraries

import matplotlib.pyplot as plt
import cv2
import pandas
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LeakyReLU
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D,Conv2DTranspose
from keras.layers.core import Flatten
from keras.optimizers import RMSprop,Adam,SGD
import numpy as np
import keras
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras.layers import *
from keras.models import model_from_json
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os
import scipy.misc
from glob import glob

#Create a de raining class
class IDGAN():
  
  def __init__(self):

    self.img_rows = 256 #No of rows in image after resize
    self.img_cols = 256 #No of columns in image after resize
    self.channels = 3   #No of image channels

    self.img_shape = (self.img_rows, self.img_cols, self.channels) #Image Shape
    
    self.dataset_name = 'rain' # Name of the Dataset
    self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols)) #Loading the data from the data_loader.py
    
    self.disc_out = (14, 14, 72) #Output of the Multi Scale Discriminator to incorporate Global context of Image 
  
    self.discriminator = self.build_discriminator() # Bulid the Discriminator
    self.generator = self.build_generator() # Build the Generator
    
    self.CGAN_model = self.build_CGAN() # Build the combined GAN Network
    
    self.optimizer_cgan = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08) #Using Adam optimizer for Generator with Learning rate acc to Paper

    self.optimizer_discriminator = SGD(lr=1E-3, momentum=0.9, decay=1e-6, nesterov=False) #Using SGD for discriminator
    
  def build_CGAN(self):
    
    self.discriminator.trainable = False # During training of Generator stop Discriminator

       
    img_B = Input(shape=self.img_shape) 
    fake_A = self.generator(img_B) # Fake Image generated from generator
   
    
    discriminator_output = self.discriminator([fake_A, img_B])

    CGAN_model = Model(inputs = [img_B],
                        outputs = [fake_A, fake_A, discriminator_output],
                        name = 'CGAN') # 3 Outputs for 3 losses 


    return CGAN_model    
  
  
  def build_discriminator(self):
    
    def d_layer(layer_input, filters, f_size=4, bn=True): #Discriminator Layer
         
      x = Conv2D(filters, kernel_size=f_size, strides=1)(layer_input)
      x = PReLU()(x)
      if bn:
          x = BatchNormalization(momentum=0.8)(x)
      x = MaxPooling2D()(x)
      return x
    
    def Deconv2d(layer_input, filters, kernel=4, dropout_rate=0): # Deconvolution Layer
      x = UpSampling2D(size=2)(layer_input)
      x = Conv2D(filters, kernel_size=kernel, strides=1, padding='same', activation='relu')(x)
      if dropout_rate:
        x = Dropout(dropout_rate)(x)
      x = BatchNormalization(momentum=0.8)(x)
      return x
    
    def Pyramid_Pool(layer_input): # Spatial Pyramid Pooling
      x_list = [layer_input]
      
      def Pool(size):
        x = MaxPooling2D(pool_size=(size*2,size*2))(layer_input)
        for i in range(size):
          x = Deconv2d(x,2)
        return x
      
      x_list.append(Pool(1)) # First level of Pyramid
      
      x2 = MaxPooling2D(pool_size=(4,4))(layer_input) # Second level of Pyramid
      x2 = Deconv2d(x2,2)
      x2 = Deconv2d(x2,2)
      x2 = ZeroPadding2D(padding=(1,1))(x2)
      x_list.append(x2)
      
      x3 = MaxPooling2D(pool_size=(8,8))(layer_input) # Last level of Pyramid
      x3 = Deconv2d(x3,4)
      x3 = Deconv2d(x3,4)
      x3 = Deconv2d(x3,4)
      x3 = ZeroPadding2D(padding=(3,3))(x3)
      x_list.append(x3)
      
      
      x = Concatenate(axis=-1)(x_list)
      return x
      
    
    img_A = Input(shape=self.img_shape)
    img_B = Input(shape=self.img_shape)
    combined_imgs = Concatenate(axis=-1)([img_A, img_B])
    
    x0 = d_layer(combined_imgs,64,3)
    x1 = d_layer(x0,256,3)
    x2 = d_layer(x1,512,3)
    x3 = d_layer(x2,64,3)
    x4 = Pyramid_Pool(x3)
    out = Activation('sigmoid')(x4) # Output is 72 channel for multi scale discriminator
    
    return Model([img_A,img_B],out)
  
  def build_generator(self):

      def Conv2d(layer_input,no_filters,kernel,stride,bn=False,padding='valid'): # Generator Convolution Layer
        x = Conv2D(filters=no_filters,kernel_size=kernel,strides=stride,padding=padding)(layer_input)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation('relu')(x)
        return x
      
      def dense_block(layer_input,num_layers): # Dense Block from Dense net Model using Skip Connections
        x_list = [layer_input]
        for i in range(num_layers):
          x = Conv2D(filters=32,kernel_size=(3,3),padding='same')(layer_input)
          x = BatchNormalization()(x)
          x = LeakyReLU()(x)
          x_list.append(x)
          x = Concatenate(axis=-1)(x_list) #Concatenating all skip connections
        return x
      
      def Deconv2d(layer_input, filters, kernel=4, dropout_rate=0): # UpSampling block
        x = UpSampling2D(size=2)(layer_input)
        x = Conv2D(filters, kernel_size=kernel, strides=1, padding='same', activation='relu')(x)
        if dropout_rate:
          x = Dropout(dropout_rate)(x)
        x = BatchNormalization(momentum=0.8)(x)
        return x
      
      inp=Input(shape=self.img_shape)
      
      #DownSampling
      
      x0 = Conv2d(inp,64,(3,3),(1,1),bn=True)
      x0 = MaxPooling2D()(x0)
      
      x1 = dense_block(x0,4)
      x1 = Conv2d(x1,128,(3,3),(2,2),bn=True)
      
      x2 = dense_block(x1,6)
      x2 = Conv2d(x2,256,(3,3),(2,2),bn=True)
      
      x3 = dense_block(x2,8)
      x3 = Conv2d(x3,512,(3,3),(1,1),bn=True,padding='same')
      
      x4 = dense_block(x3,8)
      x4 = Conv2d(x4,128,(3,3),(1,1),bn=True,padding='same')
      
      #UpSampling
      
      x5 = dense_block(x4,6)
      x5 = Deconv2d(x5,120)
      
      x6 = dense_block(x5,4)
      x6 = Deconv2d(x6,64)
      
      x7 = dense_block(x6,4)
      x7 = Deconv2d(x7,64)
      
      x8 = dense_block(x7,4)
      x8 = Conv2d(x8,16,(3,3),(1,1),bn=True,padding='same')
      
      x9 = ZeroPadding2D(padding=(5,5))(x8)
      
      x10 = Conv2D(filters=3,kernel_size=(3,3))(x9)
      out = Activation('tanh')(x10)
      
      return Model(inp,out)
    

  def train(self, epochs, batch_size=5, sample_interval=28):
   
    def perceptual_loss(img_true, img_generated): # Perceptual Loss as mentioned in paper using pretrained VGG16
      image_shape = self.img_shape
      vgg = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
      loss_block3 = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
      loss_block3.trainable = False
      loss_block2 = Model(inputs=vgg.input, outputs=vgg.get_layer('block2_conv2').output)
      loss_block2.trainable = False
      loss_block1 = Model(input=vgg.input, outputs = vgg.get_layer('block1_conv2').output)
      loss_block1.trainable = False
      return K.mean(K.square(loss_block1(img_true) - loss_block1(img_generated))) + 2*K.mean(K.square(loss_block2(img_true) - loss_block2(img_generated))) + 5*K.mean(K.square(loss_block3(img_true) - loss_block3(img_generated)))
    
    self.discriminator.trainable = False # Set the Discriminator to false for training Generator
    
    self.generator.compile(loss=perceptual_loss , optimizer= self.optimizer_cgan) #Compile the Generator
    
    CGAN_loss = ['mae', perceptual_loss, 'mse'] #All three Loses
    CGAN_loss_weights = [6.6e-3, 1 , 6.6e-3]
    self.CGAN_model.compile(loss = CGAN_loss, loss_weights = CGAN_loss_weights,
                            optimizer = self.optimizer_cgan)

    #To train the Discriminator set trainable to true
    self.discriminator.trainable = True
    self.discriminator.compile(loss="mse",
                                    optimizer = self.optimizer_discriminator)
    start_time = datetime.datetime.now()
    
    valid = np.ones((batch_size,) + self.disc_out) # For Real world Images
    fake = np.zeros((batch_size,) + self.disc_out) # For Generated Images
    
    for epoch in range(epochs):
      for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):
        fake_A = self.generator.predict(imgs_B) # Generated Image
  
        
        d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
        d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) #Discriminator Loss
  
        self.CGAN_model.trainable = True # Train the Combined model
        self.discriminator.trainable = False
        g_loss = self.CGAN_model.train_on_batch(imgs_B, [imgs_A,imgs_A,valid])
  
        elapsed_time = datetime.datetime.now() - start_time
    
        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] time: %s" % (epoch, epochs,
                                                                batch_i, self.data_loader.n_batches,
                                                                d_loss,
                                                                g_loss[0],
                                                                elapsed_time))

        
        if batch_i % sample_interval == 0:
                self.sample_images(epoch, batch_i)
    
    # Save all models after sample_interval here 25
    com_model_json = self.CGAN_model.to_json()
    gen_model_json = self.generator.to_json()
    dis_model_json = self.discriminator.to_json() 
    with open("./saved_models/com_model.json", "w") as json_file:
        json_file.write(com_model_json)
    with open("./saved_models/gen_model.json", "w") as json_file:
        json_file.write(gen_model_json)
    with open("./saved_models/dis_model.json", "w") as json_file:
        json_file.write(dis_model_json)	
    
    self.combined.save_weights("./saved_models/com_model.h5")
    self.generator.save_weights("./saved_models/gen_model.h5")
    self.discriminator.save_weights("./saved_models/dis_model.h5")
    print("Model saved")
  
  def sample_images(self, epoch, batch_i):

      #Sample Images saved after sample interval epochs
      
      os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
      r, c = 3, 3
  
      imgs_A, imgs_B = self.data_loader.load_data(batch_size=3, is_testing=True)
      fake_A = self.generator.predict(imgs_B)
  
      gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])
  
      # Rescale images 0 - 1
      gen_imgs = 0.5 * gen_imgs + 0.5
  
      titles = ['WithRain', 'Generated', 'Original']
      fig, axs = plt.subplots(r, c)
      cnt = 0
      for i in range(r):
          for j in range(c):
              axs[i,j].imshow(gen_imgs[cnt])
              axs[i, j].set_title(titles[i])
              axs[i,j].axis('off')
              cnt += 1
      fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
      plt.close()


#Training
gan=IDGAN()
## Train the model
gan.train(epochs=150, batch_size=1, sample_interval=25)

#Testing
## use the trained model to generate data
test_model = gan.build_generator()
test_model.load_weights("./saved_models/gen_model.h5")
path = glob("./dataset/rain/test_nature/*")
num = 1
for img in path:
    img_B = scipy.misc.imread(img, mode='RGB').astype(np.float)
    m,n,d = img_B.shape
    img_show = np.zeros((m,2*n,
    img_b = np.array([img_B])/127.5 - 1
    fake_A = 0.5* (test_model.predict(img_b))[0]+0.5
    img_show[:,:n,:] = img_B/255
    img_show[:,n:2*n,:] = fake_A
    scipy.misc.imsave("./images/rain/test_nature/%d.jpg" % num,img_show)
    num = num + 1 