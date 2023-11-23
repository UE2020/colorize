# Colorize!
A deep learning image &amp; video colorizer using Rust and libtorch.

## Training
To initialize the model, you'll need to run `src/transform.py` and `src/model.py` to initialize the LAB<->RGB and pre-trained generator torchscripts, respectively.
This requires PyTorch and the fastai library.

Training is as simple as running with the following arguments, where use_gan is a boolean argument:
```
./target/release/autoencoder train starting_model.pt /data/path duration_in_hours use_gan
```

See below for pre-trained model.

### Obtaining a dataset
The ImageNet Object Localization Challenge dataset (a subset of the full ImageNet dataset) is available on Kaggle,
and was used to train the baseline model. A diverse sampling of images is recommended to avoid overfitting.

Any dataset that consists of images in a folder is usable, as long as there are no corrupted images or non-image files. Subdirectories will be crawled automatically.

### 3-Step Training Procedure
Models are trained in three steps to reduce the undesirable visual artifacts caused by GAN training:
1. Train for a long time without the discriminator network (use_gan = false).
2. Continue training the network produced by the previous step for a shorter time with the discriminator network enabled (use_gan = true).
3. **Merge** the two resulting networks using the pre-defined weighted average formula: `./target/release/autoencoder merge gan.pt no_gan.pt` (order matters). The merged model will be saved to `./merged.pt`, beware of overwriting any model that may have already been there.

## Running
Running the model is as simple as:
```
./target/release/autoencoder test model.pt image.jpg image_size
```
Images will be written to `./fixed.jpg`.
Only powers of 2 may be used for the image_size parameter, although 256 is recommended, 512 and 1024 are useful for colorizing fine details.

A pre-trained model is available here: 
https://drive.google.com/file/d/1S6wAA-YkJsOVdh5-oHC6DkyPvfWiACA7/view?usp=sharing

## Demo

Colorizing legacy photos:
<img src="https://i.imgur.com/O0Lhm75.jpeg" width="400">
<img src="https://i.imgur.com/MefnRvW.jpeg" width="400">
<img src="https://i.imgur.com/ly1q00t.jpeg" width="400">
<img src="https://i.imgur.com/sulnni9.jpeg" width="400">
