### -*- coding:utf-8 -*-

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Reshape, merge, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Convolution2D

from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Flatten, Dropout

import math
import numpy as np

import os
from keras.datasets import mnist
from keras.optimizers import Adam
from PIL import Image

BATCH_SIZE = 32
NUM_EPOCH = 50
CLASS_NUM = 10

class CGAN():
 
    def __init__(self):
        #self.path = '/volumes/data/dataset/gan/cgan_generated_images/'
        self.path = 'images/'
        #mnistデータ用の入力データサイズ
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # 潜在変数の次元数 
        self.z_dim  =1000
        
        self.n_critic = 5

        # 画像保存の際の列、行数
        self.row = 5
        self.col = 5
        self.row2 = 1 # 連続潜在変数用
        self.col2 = 10# 連続潜在変数用 

        
        # 画像生成用の固定された入力潜在変数
        self.noise_fix1 = np.random.normal(0, 1, (self.row * self.col, self.z_dim)) 
        # 連続的に潜在変数を変化させる際の開始、終了変数
        self.noise_fix2 = np.random.normal(0, 1, (1, self.z_dim))
        self.noise_fix3 = np.random.normal(0, 1, (1, self.z_dim))

        # 横軸がiteration数のプロット保存用np.ndarray
        self.g_loss_array = np.array([])
        self.d_loss_array = np.array([])
        self.d_accuracy_array = np.array([])
        self.d_predict_true_num_array = np.array([])
        self.c_predict_class_list = []

        self.discriminator_optimizer = Adam(lr=1e-5, beta_1=0.1)
        self.combined_optimizer = Adam(lr=1e-4, beta_1=0.5, beta_2=0.9)

        # discriminatorモデル
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', 
            optimizer=self.discriminator_optimizer,
            metrics=['accuracy'])

        # Generatorモデル
        self.generator = self.build_generator()
        # generatorは単体で学習しないのでコンパイルは必要ない
        #self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        self.combined = self.build_combined()
        #self.combined = self.build_combined2()
        self.combined.compile(loss='binary_crossentropy', optimizer=self.combined_optimizer)

        # Classifierモデル
        #self.classifier = self.build_classifier()



    def build_generator(self):
        model = Sequential()
        model.add(Dense(input_dim=(self.z_dim + CLASS_NUM), output_dim=1024)) # z=100, y=10
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(128*7*7))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Reshape((7,7,128), input_shape=(128*7*7,)))
        model.add(UpSampling2D((2,2)))
        model.add(Convolution2D(64,5,5,border_mode='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((2,2)))
        model.add(Convolution2D(1,5,5,border_mode='same'))
        model.add(Activation('tanh'))
        return model

    def build_discriminator(self):
 .       model = Sequential()
        model.add(Convolution2D(64,5,5,\
              subsample=(2,2),\
              border_mode='same',\
              input_shape=(self.img_rows,self.img_cols,(1+CLASS_NUM))))
        model.add(LeakyReLU(0.2))
        model.add(Convolution2D(128,5,5,subsample=(2,2)))
        model.add(LeakyReLU(0.2))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        return model

    def build_combined(self):
        z = Input(shape=(self.z_dim,))
        y = Input(shape=(CLASS_NUM,))
        img_10 = Input(shape=(self.img_rows,self.img_cols,CLASS_NUM,))
        z_y = merge([z, y],mode='concat',concat_axis=-1)
      
        img = self.generator(z_y) # [batch, WIDTH, HEIGHT, channel=1]
        img_11 = merge([img, img_10],mode='concat', concat_axis=3)
        self.discriminator.trainable= False
        valid = self.discriminator(img_11)
        model = Model(input = [z, y, img_10], output = valid)
        return model

    def combine_images(self,generated_images):
        total = generated_images.shape[0]
        cols = int(math.sqrt(total))
        rows = int(math.ceil(float(total)/cols))
        WIDTH, HEIGHT = generated_images.shape[1:3]
        combined_image = np.zeros((HEIGHT*rows, WIDTH*cols),
                    dtype=generated_images.dtype)
      
        for index, image in enumerate(generated_images):
            i = int(index/cols)
            j = index % cols
            combined_image[WIDTH*i:WIDTH*(i+1), HEIGHT*j:HEIGHT*(j+1)] = image[:, :,0]
        return combined_image

    def label2images(self,label):
        images = np.zeros((self.img_rows,self.img_cols,CLASS_NUM))
        images[:,:,label] += 1
        return images
  
    def label2onehot(self,label):
        onehot = np.zeros(CLASS_NUM)
        onehot[label] = 1
        return onehot

    def train(self):
        (X_train, y_train), (_, _) = mnist.load_data()
        X_train = (X_train.astype(np.float32) - 127.5)/127.5
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2],1)
      
        discriminator = self.build_discriminator()
        d_opt = Adam(lr=1e-5, beta_1=0.1)
        discriminator.compile(loss='binary_crossentropy', optimizer=d_opt, metrics=['accuracy'])
      
        # generator+discriminator （discriminator部分の重みは固定）
        discriminator.trainable = False
        generator = self.build_generator()
        combined = self.build_combined()
      
        g_opt = Adam(lr=.8e-4, beta_1=0.5)
        combined.compile(loss='binary_crossentropy', optimizer=g_opt)

        # 学習結果を格納
        self.g_loss_array = np.zeros(epochs)
        self.d_loss_array = np.zeros(epochs)
        self.d_accuracy_array = np.zeros(epochs)
        self.d_predict_true_num_array = np.zeros(epochs)

        num_batches = int(X_train.shape[0] / BATCH_SIZE)
        print('Number of batches:', num_batches)

        ### 学習開始
        for epoch in range(NUM_EPOCH):
      
            for index in range(num_batches):
                # generator用データ整形
                noise_z = np.array([np.random.uniform(-1, 1, self.z_dim) for _ in range(BATCH_SIZE)])
                noise_y_int = np.random.randint(0,CLASS_NUM,BATCH_SIZE) # label番号を生成する乱数,BATCH_SIZE長
                noise_y = np.array([self.label2onehot(i) for i in noise_y_int]) #shape[0]:batch, shape[1]:class
                noise_z_y = np.concatenate((noise_z, noise_y),axis=1) # zとyを結合
                f_img = generator.predict(noise_z_y, verbose=0)
                f_img_10 = np.array([self.label2images(i) for i in noise_y_int]) # 生成データラベルの10ch画像
                f_img_11 = np.concatenate((f_img, f_img_10),axis=3)
      
                # discriminator用データ整形
                r_img = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE] # 実データの画像
                label_batch = y_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE] # 実データのラベル
                r_img_10 = np.array([self.label2images(i) for i in label_batch]) # 実データラベルの10ch画像
                r_img_11 = np.concatenate((r_img, r_img_10),axis=3)
      
                ''' 
                # 生成画像を出力
                if index % 500 == 0:
                    image = combine_images(generated_images)
                    image = image*127.5 + 127.5
                    if not os.path.exists(self.path):
                        os.mkdir(self.path)
                    Image.fromarray(image.astype(np.uint8))\
                        .save(self.path+"%04d_%04d.png" % (epoch, index))
                '''
                # 生成画像を出力
                if index % 500 == 0:
                    noise = np.array([np.random.uniform(-1, 1, self.z_dim) for _ in range(BATCH_SIZE)])
                    randomLabel_batch = np.arange(BATCH_SIZE)%10  # label番号を生成する乱数,BATCH_SIZE長
                    randomLabel_batch_onehot = np.array([self.label2onehot(i) for i in randomLabel_batch]) #shape[0]:batch, shape[1]:class
                    noise_with_randomLabel = np.concatenate((noise, randomLabel_batch_onehot),axis=1) # zとyを結合
                    generated_images = generator.predict(noise_with_randomLabel, verbose=0)
                    image = self.combine_images(generated_images)
                    image = image*127.5 + 127.5
                    if not os.path.exists(self.path):
                        os.mkdir(self.path)
                    Image.fromarray(image.astype(np.uint8))\
                        .save(self.path+"%04d_%04d.png" % (epoch, index))
      
      
                # discriminatorを更新
                X = np.concatenate((r_img_11, f_img_11))
                y = [1]*BATCH_SIZE + [0]*BATCH_SIZE
                y = np.array(y)
                #print(y.shape)
                d_loss = discriminator.train_on_batch(X, y)
      
                # generatorを更新
                noise = np.array([np.random.uniform(-1, 1, self.z_dim) for _ in range(BATCH_SIZE)])
                randomLabel_batch = np.random.randint(0,CLASS_NUM,BATCH_SIZE) # label番号を生成する乱数,BATCH_SIZE長
                randomLabel_batch_onehot = np.array([self.label2onehot(i) for i in randomLabel_batch]) #shape[0]:batch, shape[1]:class
                randomLabel_batch_image = np.array([self.label2images(i) for i in randomLabel_batch]) # 生成データラベルの10ch画像
                g_loss = combined.train_on_batch([noise, randomLabel_batch_onehot, randomLabel_batch_image], np.array([1]*BATCH_SIZE))
                print("epoch: %d, batch: %d, g_loss: %f, d_loss: %f" % (epoch, index, g_loss, d_loss))

            # np.ndarrayにloss関数を格納
            self.g_loss_array[epoch] = g_loss0
            self.d_loss_array[epoch] = d_loss[0]
            self.d_accuracy_array[epoch] = 100. * d_loss[1]
  
            generator.save_weights('generator.h5')
            discriminator.save_weights('discriminator.h5')

if __name__ == '__main__':
    gan = CGAN()
    gan.train()
