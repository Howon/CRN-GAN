Barbara Zhan  bz2310 

Pooja Kathail   pk2485

Howon Byun   hb2458

## Data
First run 
```
python download_models.py
```
to download VGG weights used for loss calculation in Cascaded Refinement Network. 
Goal of this project is to remove this dependency and replace it with the discriminator from Pix2Pix.

Then, two datasets from [Cityscapes](https://www.cityscapes-dataset.com/) need to be downloaded.
1. gtFine_trainvaltest.zip, which holds semantic labels that are used as the training inputs.
2. leftImg8bit_trainvaltest.zip used as training targets.

Once these are downloaded and unzipped in this directory, run resize.m by doing
```
cat resize.m| matlab -nodesktop -nosplash
```
This Matlab script will resize images in training and validations directories from both gtFine and leftImg8bit
folders into data/cityscapes/semantics and data/cityscapes/images, respectively.

Once this is done, simply perform
```
python crn.py
```

## Citations
[Photographic Image Synthesis with Cascaded Refinement Networks](http://cqf.io/ImageSynthesis/)

[Image-to-Image Translation with Conditional Adversarial Networks/Pix2Pix](https://arxiv.org/pdf/1611.07004v1.pdf)

[Pix2Pix Tensorflow Implementation](https://github.com/affinelayer/pix2pix-tensorflow)
