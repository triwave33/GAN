### -*-coding:utf-8-*-
from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model

from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys

import numpy as np

class DCGAN():
    
    def __init__(self):
        self.path = "./images"

        # datasize for mnist 
        self.img_rows = 28 
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.z_dim = 5
        
        ########## settings for image saving ###########
        # for image saving
        self.row = 5
        self.col = 5
        self.row2 = 1 # for latent space
        self.col2 = 10# for latent space
        
        # as a noise "seed" for creating images from the same value
        self.noise_fix1 = np.random.normal(0, 1, (self.row * self.col, self.z_dim)) 
        # for moving latent variable (z) from fix2 to fix3
        self.noise_fix2 = np.random.normal(0, 1, (1, self.z_dim))
        self.noise_fix3 = np.random.normal(0, 1, (1, self.z_dim))

        ###############################################

        self.g_loss_array = np.array([])
        self.d_loss_array = np.array([])
        self.d_accuracy_array = np.array([])
        self.d_predict_true_num_array = np.array([])
        self.c_predict_class_list = []

        discriminator_optimizer = Adam(lr=1e-5, beta_1=0.1)
        combined_optimizer = Adam(lr=2e-4, beta_1=0.5)

        # discriminator model
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', 
            optimizer=discriminator_optimizer,
            metrics=['accuracy'])

        # Generator model
        self.generator = self.build_generator()

        self.combined = self.build_combined1()
        #self.combined = self.build_combined2()
        self.combined.compile(loss='binary_crossentropy', optimizer=combined_optimizer)

        # Classifier model
        self.classifier = self.build_classifier()

    def build_generator(self):

        noise_shape = (self.z_dim,)
        model = Sequential()
        model.add(Dense(1024, input_shape=noise_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(128*7*7))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Reshape((7,7,128), input_shape=(128*7*7,)))
        model.add(UpSampling2D((2,2)))
        model.add(Conv2D(64,5,5,padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((2,2)))
        model.add(Conv2D(1,5,5,padding='same'))
        model.add(Activation('tanh'))
        model.summary()
        return model

    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)
        
        model = Sequential()
        model.add(Conv2D(64,5,5, strides=(2,2),\
                  padding='same', input_shape=img_shape))
        model.add(LeakyReLU(0.2))
        model.add(Conv2D(128,5,5, strides=(2,2)))
        model.add(LeakyReLU(0.2))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))   
        return model
    
    def build_combined1(self):
        self.discriminator.trainable = False
        model = Sequential([self.generator, self.discriminator])
        return model

    def build_combined2(self):
        z = Input(shape=(self.z_dim,))
        img = self.generator(z)
        self.discriminator.trainable = False
        valid = self.discriminator(img)
        model = Model(z, valid)
        model.summary()
        return model
    
    # load extra model to classify the images created by generator
    def build_classifier(self):
        model = load_model("cnn_model.h5")
        model.load_weights('cnn_weight.h5')
        return model



    def train(self, epochs, batch_size=128, save_interval=50):

        # load mnist data
        (X_train, _), (_, _) = mnist.load_data()

        # normalization
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)

        num_batches = int(X_train.shape[0] / half_batch)
        print('Number of batches:', num_batches)
                                
        self.g_loss_array = np.zeros(epochs)
        self.d_loss_array = np.zeros(epochs)
        self.d_accuracy_array = np.zeros(epochs)
        self.d_predict_true_num_array = np.zeros(epochs)

        for epoch in range(epochs):
            for iteration in range(num_batches):

                # ---------------------
                #  learn Discriminator
                # ---------------------

                # generate images (half batch size) from generato
                noise = np.random.normal(0, 1, (half_batch, self.z_dim))
                gen_imgs = self.generator.predict(noise)


                # pickup images (half batch size) from dataset
                idx = np.random.randint(0, X_train.shape[0], half_batch)
                imgs = X_train[idx]

                # learn discriminator
                d_loss_real = self.discriminator.train_on_batch(
                                    imgs, np.ones((half_batch, 1)))
                d_loss_fake = self.discriminator.train_on_batch(
                                    gen_imgs, np.zeros((half_batch, 1)))
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # predict (half: fake images, half: real images)
                d_predict = self.discriminator.predict_classes(
                                np.concatenate([gen_imgs,imgs]), verbose=0)
                d_predict = np.sum(d_predict)

                # label prediction by classifier
                c_predict = self.classifier.predict_classes(
                                np.concatenate([gen_imgs,imgs]), verbose=0)


                # ---------------------
                #  learn Generator
                # ---------------------

                noise = np.random.normal(0, 1, (batch_size, self.z_dim))
                # label must be set to 1 for generator learning
                valid_y = np.array([1] * batch_size)

                # Train the generator
                g_loss = self.combined.train_on_batch(noise, valid_y)

                # progress
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

                self.g_loss_array[epoch] = g_loss
                self.d_loss_array[epoch] = d_loss[0]
                self.d_accuracy_array[epoch] = 100*d_loss[1]
                self.d_predict_true_num_array[epoch] = d_predict
                self.c_predict_class_list.append(c_predict)

            if epoch % save_interval == 0:
                
                # save images from random seed
                self.save_imgs(self.row, self.col, epoch, '', noise)
                # save images from fixed seed
                self.save_imgs(self.row, self.col, epoch, 'fromFixedValue', self.noise_fix1)
                # save transition images between two latent variables
                total_images = self.row*self.col
                noise_trans = np.zeros((total_images, self.z_dim))
                for i in range(total_images):
                    t = (i*1.)/((total_images-1)*1.)
                    noise_trans[i,:] = t * self.noise_fix2 + (1-t) * self.noise_fix3
                self.save_imgs(self.row2, self.col2, epoch, 'trans', noise_trans)

                # discriminate images generated from Generator (10000 samples)
                noise = np.random.normal(0, 1, (10000, self.z_dim))
                class_res = self.classifier.predict_classes(self.generator.predict(noise), verbose=0)
                # plot histgram
                plt.hist(class_res)
                plt.savefig(self.path + "mnist_hist_%d.png" % epoch)
                plt.ylim(0,2000)
                plt.close()


                # plot learning result
                fig, ax = plt.subplots(4,1, figsize=(8.27,11.69))
                ax[0].plot(self.g_loss_array[:epoch])
                ax[0].set_title("g_loss")
                ax[1].plot(self.d_loss_array[:epoch])
                ax[1].set_title("d_loss")
                ax[2].plot(self.d_accuracy_array[:epoch])
                ax[2].set_title("d_accuracy")
                ax[3].plot(self.d_predict_true_num_array[:epoch])
                ax[3].set_title("d_predict_true_num_array")
                fig.suptitle("epoch: %5d" % epoch)
                fig.savefig(self.path + "training_%d.png" % epoch)
                plt.close()

        # save weights
        self.generator.save_weights(self.path + "generator_%s.h5" % epoch)
        self.discriminator.save_weights(self.path + "discriminator_%s.h5" % epoch)


            

    def save_imgs(self, row, col, epoch, filename, noise):
    
        gen_imgs = self.generator.predict(noise)
    
        # rescall generated images
        gen_imgs = 0.5 * gen_imgs + 0.5
    
    
        fig, axs = plt.subplots(row, col)
        cnt = 0
        if row == 1:
            for j in range(col):
                axs[j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[j].axis('off')
                cnt += 1
        else:
            for i in range(row):
                for j in range(col):
                    axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                    axs[i,j].axis('off')
                    cnt += 1

        fig.suptitle("epoch: %5d" % epoch)
        fig.savefig(self.path + "mnist_%s_%d.png" % (filename, epoch))
        plt.close()
    
 
if __name__ == '__main__':
    gan = DCGAN()
    gan.train(epochs=100000, batch_size=32, save_interval=1000)






