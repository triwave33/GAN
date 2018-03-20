import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import glob

# set working dir

images = []
files = glob.glob("draw_flag3*.png")

for f in files:
	im = Image.open(f)
	images.append(im)

images[0].save('pillow_imagedraw.gif',\
               save_all=True, append_images=images[1:],\
			   optimize=True, duration=500, loop=0)
