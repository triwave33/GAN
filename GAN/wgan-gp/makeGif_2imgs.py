import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import glob

# set working dir
wd1 = "gifFolder/im1/"
wd2 = "gifFolder/im2/"
wd_out = "gifFolder/im_out/"

images = []
files1 = glob.glob(wd1 + "*_fromFixedValue*.png")
files2 = glob.glob(wd2 + "*_fromFixedValue*.png")

for f1, f2 in zip(files1, files2):
    im1 = Image.open(f1)
    im2 = Image.open(f2)
    canvas = Image.new("L", (1300,480), (255))
    canvas.paste(im1,(0,0))
    canvas.paste(im2,(660,0))

    images.append(canvas)

images[0].save(wd_out + 'pillow_imagedraw.gif',\
               save_all=True, append_images=images[1:],\
               optimize=False, duration=300, loop=0)
