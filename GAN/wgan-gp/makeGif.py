import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import glob

# set working dir
wd = "gifFolder/im1/"

images = []
files = glob.glob(wd + "*_fromFixedValue*.png")

for f in files:
	im = Image.open(f)
	images.append(im)

images[0].save("gifFolder/im_out/" + 'pillow_imagedraw.gif',\
               save_all=True, append_images=images[1:],\
			   optimize=False, duration=200, loop=0)
