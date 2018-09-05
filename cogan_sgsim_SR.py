#alexsun
#date: 06242018
#Modified after the Keras discogan implementation from
#https://raw.githubusercontent.com/eriklindernoren/Keras-GAN/master/discogan/discogan.py
#This is used to generate Figures 1, 2 of the GRL paper
#rev: 07042018
#this is trained on TACC Maverick, $WORK/cogan/config1_a
#=========================================================================================
from __future__ import print_function, division

import scipy
import random as rn
import tensorflow as tf
from numpy.random import seed

seed(1111)
rn.seed(1989)
tf.set_random_seed(1989)
#why do I need this?
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import sklearn.preprocessing as skp
import seaborn as sns

import scipy

from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import sys


import numpy as np
import os
from skimage.measure import compare_ssim, compare_nrmse
from mpl_toolkits.axes_grid1 import make_axes_locatable

#disable keras warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings

#image dimension
N = 128
#kernel dimension
KN_WIDTH = 4

class SPIDGAN():
    def __init__(self, imshape):
        # Input shape
        self.img_rows, self.img_cols = imshape
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.heads=None

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 16)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64
        #this is the setup from the original Kim paper
        optimizer = Adam(0.0002, 0.5, 0.999)
        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.d_B.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Build the generators
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Objectives
        # + Adversarial: Fool domain discriminators
        # + Translation: Minimize MAE between e.g. fake B and true B
        # + Cycle-consistency: Minimize MAE between reconstructed images and original
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[ valid_A, valid_B,
                                        fake_B, fake_A,
                                        reconstr_A, reconstr_B ])
        #print (self.combined.summary())
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                              optimizer=optimizer)
        
    def build_generator(self):
        """U-Net Generator"""
        def conv2d(layer_input, filters, f_size=KN_WIDTH, normalize=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalize:
                d = InstanceNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=KN_WIDTH, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)

            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, normalize=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)
        d6 = conv2d(d5, self.gf*8)
        d7 = conv2d(d6, self.gf*8)

        # Upsampling
        u1 = deconv2d(d7, d6, self.gf*8)
        u2 = deconv2d(u1, d5, self.gf*8)
        u3 = deconv2d(u2, d4, self.gf*8)
        u4 = deconv2d(u3, d3, self.gf*4)
        u5 = deconv2d(u4, d2, self.gf*2)
        u6 = deconv2d(u5, d1, self.gf)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(self.channels, kernel_size=KN_WIDTH, strides=1,
                            padding='same', activation='tanh')(u7)
        dd = Model(d0, output_img)
        #print (dd.summary())
        return dd

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=KN_WIDTH, normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = Input(shape=self.img_shape)
        #switch for super-resolution
        SR = False
        if SR:
            d1 = d_layer(img, self.df, normalization=False)
            d1 = d_layer(d1, self.df, stride_size=2)
            #128
            d2 = d_layer(d1, self.df*2)
            d2 = d_layer(d2, self.df*2, stride_size=2)
            #256
            d3 = d_layer(d2, self.df*4)
            d3 = d_layer(d3, self.df*4, stride_size=2)
            #512    
            d4 = d_layer(d3, self.df*8)
            d4 = d_layer(d4, self.df*8, stride_size=2)
        else:
            d1 = d_layer(img, self.df, normalization=False)
            d2 = d_layer(d1, self.df*2)
            d3 = d_layer(d2, self.df*4)
            d4 = d_layer(d3, self.df*8)
            
        validity = Conv2D(1, kernel_size=KN_WIDTH, strides=1, padding='same')(d4)

        dd = Model(img, validity)
        print (dd.summary())

        return dd

    def mytrain(self, epochs, batch_size=20, sample_interval=50, reTrain=True):
        def getBatch():
            X1, X2 = self.loadData(task='train')
            n_batches = int(X1.shape[0] / batch_size)
            for i in range(n_batches-1):
                imgs_A = X1[i*batch_size:(i+1)*batch_size, :, :, :]
                imgs_B = X2[i*batch_size:(i+1)*batch_size, :, :, :]
                
                yield imgs_A, imgs_B 
        if reTrain:
            # Adversarial loss ground truths
            valid = np.ones((batch_size,) + self.disc_patch)
            fake = np.zeros((batch_size,) + self.disc_patch)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
        
                for epoch in range(epochs):
                    for batch_i, (imgs_A, imgs_B) in enumerate(getBatch()):
        
                        # ----------------------
                        #  Train Discriminators
                        # ----------------------
        
                        # Translate images to opposite domain
                        fake_B = self.g_AB.predict(imgs_A)
                        fake_A = self.g_BA.predict(imgs_B)
        
                        # Train the discriminators (original images = real / translated = Fake)
                        dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
                        dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                        dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)
        
                        dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                        dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                        dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)
        
                        # Total disciminator loss
                        d_loss = 0.5 * np.add(dA_loss, dB_loss)
        
                        # ------------------
                        #  Train Generators
                        # ------------------
        
                        # Train the generators
                        g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, valid, \
                                                                                 imgs_B, imgs_A, \
                                                                                 imgs_A, imgs_B])
        
        
                        # Plot the progress
                        print ("[%d] [%d/], [d_loss: %f, g_loss: %f]" % (epoch, batch_i, d_loss[0], g_loss[0]))
        
                        # If at save interval => save generated image samples
                        if batch_i % sample_interval == 0:
                            self.sample_images(epoch, batch_i)
            self.g_AB.save_weights('gan_ab.h5')
            self.g_BA.save_weights('gan_ba.h5')
        else:
            self.g_AB.load_weights('gan_ab.h5')
            self.g_BA.load_weights('gan_ba.h5')
            
    def loadData(self, task='train', idx=None):
        '''
        load training data
        g1 learn head
        g2 learn facies
        @param task, Train or Test
        '''
        XMIN = 0.0
        XMAX = 1.0

        if self.heads is None:
            #only do this once
            heads, facies = np.load('allsghead.npy')
            self.heads=heads
            #this is permeability
            self.facies = facies 

            #minmaxscaler takes min/max along each feature, i.e., each realization in columns
            self.scaler1 = skp.MinMaxScaler(feature_range=(XMIN, XMAX), copy=False)
            self.scaler1.fit_transform(self.facies)
         
            self.scaler2 = skp.MinMaxScaler(feature_range=(XMIN, XMAX), copy=False)
            self.scaler2.fit_transform(self.heads)            
        else:
            heads = self.heads
            facies = self.facies
        
        nTrain = 400
        nTest = facies.shape[1] - nTrain
        print ('number of realizations', facies.shape[1])
        #number of columns = number of realizations
        #number of rows = number of cells
                
        heads = np.transpose(heads)
        facies= np.transpose(facies)    

        if task=='train':
            # Images in domain A and B        
            X1 = np.zeros((nTrain, N, N, 1), dtype=np.float32)
            X2 = np.zeros((nTrain, N, N, 1), dtype=np.float32)
            for i in range(nTrain):
                X1[i,:,:,0] = np.reshape(facies[i,:], (N,N))
                X2[i,:,:,0] = np.reshape(heads[i,:], (N,N))
            
            return X1, X2

        else:
            # Images in domain A and B        
            X1 = np.zeros((nTest, N, N, 1), dtype=np.float32)
            X2 = np.zeros((nTest, N, N, 1), dtype=np.float32)
            for i in range(nTest):
                X1[i,:,:,0] = np.reshape(facies[nTrain+i,:], (N,N))
                X2[i,:,:,0] = np.reshape(heads[nTrain+i,:], (N, N))
            #randomly select a number from 0 to nTest
            if idx is None:
                idx = np.random.randint(0, nTest, 1)

            return X1[idx], X2[idx]

    def sample_images(self, epoch, batch_i, idx=None):
        def rescale(img, stype):
            if idx is None:
                return img
            else:
                #do rescaling
                img = np.reshape(img, (N*N,1))
                if stype=='h':
                    img = img*(self.scaler2.data_max_[idx] - self.scaler2.data_min_[idx])+self.scaler2.data_min_[idx]
                elif stype=='k':
                    img = img*(self.scaler1.data_max_[idx] - self.scaler1.data_min_[idx])+self.scaler1.data_min_[idx]
                return np.reshape(img, (1,N,N,1))
        
        datasetname = 'sgsim'
        os.makedirs('images/%s' % datasetname, exist_ok=True)
        r, c = 1, 4

        imgs_A, imgs_B = self.loadData(task='testing', idx=idx)

        # Translate images to the other domain
        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)
        # Translate back to original domain
        reconstr_A = self.g_BA.predict(fake_B)
        reconstr_B = self.g_AB.predict(fake_A)
        
        RESCALE = False
        if RESCALE:
            # scale back to the original domain
            imgs_A = rescale(imgs_A, 'k')
            imgs_B = rescale(imgs_B, 'h')
            fake_A = rescale(fake_A, 'k')
            fake_B = rescale(fake_B, 'h')
            reconstr_A = rescale(reconstr_A, 'k')
            reconstr_B = rescale(reconstr_B, 'h')
        
        #gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])
        gen_imgs = np.concatenate([imgs_A, fake_B, imgs_B, fake_A])
        print (gen_imgs.shape)
        titles = ['Original logK', 'Generated h', 'Original h', 'Generated logK']
        fig, axs = plt.subplots(r, c, figsize=(12,6), dpi=300)
        cnt = 0
        X, Y = np.meshgrid(np.arange(N), np.arange(N))
        for i in range(c):
                if (i==1 or i==2):
                    #plot head
                    if RESCALE:
                        cs = axs[i].contourf(X,Y, gen_imgs[cnt, :, :, 0], N=10)
                    else:
                        cs = axs[i].contourf(X,Y, gen_imgs[cnt, :, :, 0], vmin=-0.1, vmax=1.1,
                                             levels=np.arange(-0.2, 1.2, 0.1))                        
                else:
                    if RESCALE:
                        cs = axs[i].contourf(X,Y, gen_imgs[cnt, :, :, 0], cmap='coolwarm', N=10)
                    else:
                        cs = axs[i].contourf(X,Y, gen_imgs[cnt, :, :, 0], cmap='coolwarm',
                                               vmin=-0.1, vmax=1.1,levels=np.arange(-0.2, 1.2, 0.1))

                divider = make_axes_locatable(axs[i])
                cax = divider.append_axes("bottom", size="5%", pad=0.05) 
                cbar = plt.colorbar(cs, cax=cax, ax=axs[i], orientation="horizontal", format="%.1f")
                cbar.ax.set_xticklabels(['', '0', '0.2', '0.4', '0.6', '0.8', '1.0']) 
                cbar.ax.tick_params(labelsize=10)
                axs[i].set_aspect('equal')
                axs[i].set_title(titles[cnt])
                axs[i].axis('off')
               
                cnt += 1
        if not idx is None:
            fig.savefig("images/%s/%d_%d_%d.eps" % (datasetname, epoch, batch_i, idx))
        else:
            fig.savefig("images/%s/%d_%d.png" % (datasetname, epoch, batch_i))
        plt.close()
    
    def drawTest(self):
        for i in range(10):
            self.sample_images(epoch=0, batch_i=0, idx=np.array([i]))


    def calculateStats(self, reRun=False):
        #realizations used for testing
        nrz = 1000
        
        if reRun:
            midrowMC = np.zeros((3, nrz, N))
            midrowDL = np.zeros((3, nrz, N))
            ssimHead = np.zeros(nrz)
            mseHead = np.zeros(nrz)
            ssimK = np.zeros(nrz)
            mseK = np.zeros(nrz)
            
            for i in range(nrz):
                print ('realization ', i)
                perm, true_head = self.loadData(task='testing', idx=np.array([i]))
                # Translate perm into head
                fake_head = self.g_AB.predict(perm)
                # Translate head into perm
                fake_perm = self.g_BA.predict(true_head)
                # get the middle row
                midrowDL[0, i, :] = fake_head[0, int(0.5*N), :, 0]
                midrowMC[0, i, :] = true_head[0, int(0.5*N), :, 0]
    
                midrowDL[1, i, :] = fake_head[0, int(0.25*N), :, 0]
                midrowMC[1, i, :] = true_head[0, int(0.25*N), :, 0]
    
                midrowDL[2, i, :] = fake_head[0, int(0.75*N), :, 0]
                midrowMC[2, i, :] = true_head[0, int(0.75*N), :, 0]
                ssimHead[i] = compare_ssim(true_head[0,:,:,0], fake_head[0,:,:,0], win_size=7)
                ssimK[i] = compare_ssim(perm[0,:,:,0], fake_perm[0,:,:,0], win_size=7)
                print ('ssim=', ssimHead[i], ssimK[i])
                mseHead[i] = compare_nrmse(true_head[0,:,:,0], fake_head[0,:,:,0])
                mseK[i] = compare_nrmse(perm[0,:,:,0], fake_perm[0,:,:,0])
                print ('rmse=', mseHead[i], mseK[i]) 
            np.save('mcresults.npy', [midrowMC, midrowDL])
            np.save('ssimresults.npy', [ssimHead,ssimK,mseHead,mseK])
        else:
            midrowMC, midrowDL = np.load('mcresults.npy')
            ssimHead,ssimK,mseHead,mseK = np.load('ssimresults.npy')
        print ('mean ssim_h=', np.mean(ssimHead))
        print ('mean ssim_K=', np.mean(ssimK))
        
        #calculate variance
        fig,ax = plt.subplots(2,3, figsize=(12,12), dpi=300)
        for i in range(3):
            ax[0,i].plot(np.mean(midrowDL[i,:,:], axis=0), 'b--', label='DL')
            ax[0,i].plot(np.mean(midrowMC[i,:,:], axis=0), 'g-', label='MC')

            ax[1,i].plot(np.var(midrowDL[i,:,:], axis=0), 'b--', label='DL')
            ax[1,i].plot(np.var(midrowMC[i,:,:], axis=0), 'g-', label='MC')
        plt.legend()
        plt.savefig('headmoment.png')   
        
        fig,ax=plt.subplots(1,2, figsize=(8,4))
        sns.distplot(ssimHead, ax=ax[0])
        ax[0].set_title('SSIM h')
        sns.distplot(ssimK,ax=ax[1])
        ax[1].set_title('SSIM logK')
        plt.savefig('ssimplot.eps')               

        fig,ax=plt.subplots(1,2, figsize=(4,4))
        sns.distplot(mseHead, ax=ax[0])
        sns.distplot(mseK, ax=ax[1])
        plt.tight_layout() 
        plt.close()
        
if __name__ == '__main__':
    gan = SPIDGAN(imshape=(N,N))
    gan.mytrain(epochs=125, batch_size=10, sample_interval=25, reTrain=False)
    #this is used to generate Figure2a, use #5
    gan.drawTest()
    #this is used to generate Figure2b, reRun=True, rerun MC, False, using existing results
    gan.calculateStats(reRun=False)