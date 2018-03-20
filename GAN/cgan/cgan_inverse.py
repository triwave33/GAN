### -*-coding:utf-8 -*-
import numpy as np
from PIL import Image

import keras
from keras.models import Sequential, Model, load_model
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


model = load_model("generator_model.h5")
model.load_weights('generator_weights.h5')

class CGAN_inverse():
 
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


        # Generatorモデル
        self.generator = load_model("generator_model.h5")
        self.generator.load_weights('generator_weights.h5')
        # generatorは単体で学習しないのでコンパイルは必要ない
        #self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)
        self.degenerator = self.build_degenerator()

        turnback_opt = Adam(lr=0.8e-4, beta_1=0.5)
        self.turnback = self.build_turnback()
        self.turnback.compile(loss='mean_squared_error', optimizer=turnback_opt)

    def build_degenerator(self):
        model = Sequential()
        model.add(Convolution2D(64,5,5,\
              subsample=(2,2),\
              border_mode='same',\
              input_shape=(self.img_rows,self.img_cols,(1+CLASS_NUM))))
        model.add(LeakyReLU(0.2))
        model.add(Convolution2D(128,5,5,subsample=(2,2)))
        model.add(LeakyReLU(0.2))
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.5))
        model.add(Dense((self.z_dim + CLASS_NUM)))
        model.add(Activation('tanh'))
        return model

    def build_turnback(self):
        z = Input(shape=(self.z_dim,))
        y = Input(shape=(CLASS_NUM,))
        img_10 = Input(shape=(self.img_rows,self.img_cols,CLASS_NUM,))
        z_y = merge([z, y],mode='concat',concat_axis=-1)
      
        img = self.generator(z_y) # [batch, WIDTH, HEIGHT, channel=1]
        img_11 = merge([img, img_10],mode='concat', concat_axis=3)
        self.generator.trainable= False
        z_ = self.degenerator(img_11)
        model = Model(input = [z, y, img_10], output = z_)
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


        # 学習結果を格納
        self.turnback_loss_array = np.zeros(NUM_EPOCH)


        ### 学習開始
        for epoch in range(NUM_EPOCH):
      
            # generator用データ整形
            noise_z = np.array([np.random.uniform(-1, 1, self.z_dim) for _ in range(BATCH_SIZE)])
            noise_y_int = np.random.randint(0,CLASS_NUM,BATCH_SIZE) # label番号を生成する乱数,BATCH_SIZE長
            noise_y = np.array([self.label2onehot(i) for i in noise_y_int]) #shape[0]:batch, shape[1]:class
            noise_z_y = np.concatenate((noise_z, noise_y),axis=1) # zとyを結合
            f_img = self.generator.predict(noise_z_y, verbose=0)
            f_img_10 = np.array([self.label2images(i) for i in noise_y_int]) # 生成データラベルの10ch画像
            f_img_11 = np.concatenate((f_img, f_img_10),axis=3)

            turnback_z_y = self.turnback.predict([noise_z, noise_y,f_img_10])
      
            turnback_batch_size =8

            # 生成画像を出力
            if epoch % 500 == 0:
                noise = np.array([np.random.uniform(-1, 1, self.z_dim) for _ in range(turnback_batch_size)])
                # label番号を生成する乱数,BATCH_SIZE長
                randomLabel_batch = np.arange(turnback_batch_size)%10  
                #shape[0]:batch, shape[1]:class
                randomLabel_batch_onehot = np.array([self.label2onehot(i) for i in randomLabel_batch])
                noise_with_randomLabel = np.concatenate((noise, randomLabel_batch_onehot),axis=1) # zとyを結合
                generated_images = self.generator.predict(noise_with_randomLabel, verbose=0)
                
                randomLabel_img_10 = np.array([self.label2images(i) for i in randomLabel_batch]) # 生成データラベルの10ch画像
                turnback_noise_with_randomLabel = self.turnback.predict([noise, randomLabel_batch_onehot, randomLabel_img_10])
                turnback_generated_images = self.generator.predict(turnback_noise_with_randomLabel, verbose=0)

                image = self.combine_images(np.concatenate((generated_images,turnback_generated_images), axis=0))
                image = image*127.5 + 127.5
                if not os.path.exists(self.path):
                    os.mkdir(self.path)
                Image.fromarray(image.astype(np.uint8))\
                    .save(self.path+"%06d.png" % (epoch))
      
      
      
            # generatorを更新
            turnback_loss = self.turnback.train_on_batch([noise_z,noise_y, f_img_10], turnback_z_y)
            print("epoch: %d, turnback_loss: %f" % (epoch, turnback_loss))
            self.turnback_loss_array[epoch] = turnback_loss

  
            self.turnback.save_weights('turnback.h5')

if __name__ == '__main__':
    gan = CGAN_inverse()
    gan.train()
