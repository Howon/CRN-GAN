import imageio
import os
import sys
import numpy as np

LEFT_DIR = argv[1]
RIGHT_DIR = argv[2]
OUTPUT_DIR = argv[3]

semantics_paths = os.listdir()
images_paths = os.listdir(RIGHT_DIR)

img_paths = zip(semantics_paths, images_paths, range(len(images_paths)))

for x, y, i in img_paths:
	left_image = imageio.imread(os.path.join(LEFT_DIR, x))
	right_image = imageio.imread(os.path.join(RIGHT_DIR, y))

	imgs_comb = np.hstack(([left_image, right_image]))
	out_path = os.path.join(OUTPUT_DIR, "{}.png".format(i))
	imgs_comb_pic = imageio.imwrite(out_path, imgs_comb)
