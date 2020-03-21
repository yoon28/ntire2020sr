# Intro.

The SIA team proposed a generative adversarial network based on ESRGAN to extremely super-resolve an input image with a magnification factor of 16.

This repo is based on [ESRGAN](https://github.com/xinntao/BasicSR).

# Usage

```
python test.py -opt [yml]
```

Output files (super-resolved images) are located under `results/` folder.

Replace `[yml]` with your test settings, eg `test_x16.yml`.

In the `[yml]` file, you can designate image folder where LR images are located.Please, assign `dataroot_LQ` to your test image folder.

`pretrain_model_G` refers to the weight file, namely trained parameters.

Please leave the remaining unchanged to reproduce our results.
