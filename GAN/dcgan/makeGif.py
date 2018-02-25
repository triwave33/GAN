import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import glob

# set working dir
wd = "/volumes/data/dataset/gan/MNIST/dcgan/dcgan_generated_images/"

images = []
files = glob.glob(wd + "*_fromFixedValue*.png")

for f in files:
	im = Image.open(f)
	images.append(im)

images[0].save(wd + 'pillow_imagedraw.gif',\
               save_all=True, append_images=images[1:],\
			   optimize=False, duration=100, loop=0)
