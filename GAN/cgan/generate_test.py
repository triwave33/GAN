### -*-coding:utf-8 -*-
import numpy as np
from keras. models import Model, load_model
from PIL import Image

model = load_model("generator_model.h5")
model.load_weights('generator_weights.h5')
z_dim = 1000
CLASS_NUM = 10
REPEAT = 100

def combine_images(generated_images):
    total = generated_images.shape[0]
    cols = 10
    rows = 10
    WIDTH, HEIGHT = generated_images.shape[1:3]
    combined_image = np.zeros((HEIGHT*rows, WIDTH*cols),
    dtype=generated_images.dtype)
                        
    for index, image in enumerate(generated_images):
        i = int(index/cols)
        j = int(index % cols)
        combined_image[WIDTH*i:WIDTH*(i+1), HEIGHT*j:HEIGHT*(j+1)] = image[:, :,0]
    return combined_image

def draw_image(one_hot, filename="img"):
    noise = np.random.uniform(-1,1,[REPEAT, z_dim])
    one_hot_repeat = np.tile(one_hot, REPEAT).reshape(REPEAT, CLASS_NUM)
    noise_with_one_hot = np.concatenate([noise,one_hot_repeat],axis=1)
    generated_images = model.predict(noise_with_one_hot)
    img = combine_images(generated_images)
    #img = generated_image.reshape(28,28)
    img = img*127.5 +127.5
    Image.fromarray(img.astype(np.uint8))\
        .save("%s.png" % (filename))




# test1: "3" を描画
def test1():
    label = [3]
    one_hot = np.eye(CLASS_NUM)[label].reshape(CLASS_NUM)
    draw_image(one_hot, "draw3") 

# test2: "7" を描画
def test2():
    label = [7]
    one_hot = np.eye(CLASS_NUM)[label]
    draw_image(one_hot, "draw7") 

# test3: one_hotラベルが3,7
def test3():
    one_hot = np.array([0,0,0,1,0,0,0,1,0,0])
    draw_image(one_hot, "draw3_7") 

# test4: one_hotラベルが2,6
def test4():
    one_hot = np.array([0,0,1,0,0,0,1,0,0,0])
    draw_image(one_hot, "draw2_6") 

# test5: one_hotラベルがオール0
def test5():
    one_hot = np.zeros(10)
    draw_image(one_hot, "draw_allzero") 

# test6: one_hotラベルがオール0.5
def test6():
    one_hot = np.ones(10)*0.5
    draw_image(one_hot, "draw_all0_5") 

# test7: one_hotラベルがオール1
def test7():
    one_hot = np.ones(10)
    draw_image(one_hot, "draw_allone") 

# test8: one_hotラベルがnoise([0,1])
def test8():
    one_hot = np.random.uniform(0,1,10)
    draw_image(one_hot, "draw_allnoise") 

# test9: オール0から"3"フラグを徐々に立てていく
def test9():
    for i in np.linspace(0,10,11):
        one_hot = np.zeros(10)
        one_hot[3] = i/10.
        filename = "draw_flag3_0%d" % (i)
        draw_image(one_hot, filename)

#test1()
#test2()
#test3()
#test4()
#test5()
#test6()
#test7()
#test8()
#test9()





