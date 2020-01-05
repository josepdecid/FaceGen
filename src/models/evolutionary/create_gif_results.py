import glob

import os
from PIL import Image

path = 'D:\\USUARIS\\gonzalorecio\\CI\\checkpoints\\GA_results_2020-01-03-21-02-49.319604_2650133507'
fp_in = os.path.join(path, '*.png')
fp_out = os.path.join(path, 'evolution.gif')

img, *images = [Image.open(f) for f in sorted(glob.glob(fp_in))]
img.save(fp=fp_out, format='GIF', append_images=images,
         save_all=True, duration=200, loop=0)
