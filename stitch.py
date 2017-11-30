import imageio
import os
import sys
import numpy as np

def main():
	if len(sys.argv) != 4:
		print "Usage: python stitch.py <semantics_dir> <images_dir> <output_dir>"
		sys.exit(1)

	LEFT_DIR = sys.argv[1]
	RIGHT_DIR = sys.argv[2]
	OUTPUT_DIR = sys.argv[3]

	semantics_paths = os.listdir()
	images_paths = os.listdir(RIGHT_DIR)

	img_paths = zip(semantics_paths, images_paths, range(len(images_paths)))

	for x, y, i in img_paths:
		left_image = imageio.imread(os.path.join(LEFT_DIR, x))
		right_image = imageio.imread(os.path.join(RIGHT_DIR, y))

		imgs_comb = np.hstack(([left_image, right_image]))
		out_path = os.path.join(OUTPUT_DIR, "{}.png".format(i))
		imgs_comb_pic = imageio.imwrite(out_path, imgs_comb)

if __name__ == "__main__":
	main()
