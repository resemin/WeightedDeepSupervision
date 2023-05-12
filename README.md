# WeightedDeepSupervision

This is a proto code for revision of Artificial Intelligence in Medicine

The code consists of.

- gen_texture.py: generate a texture map from image
- gen_groundtruth.py: generate a groundtruth map from a texture and a pseudo-GT
- gen_wwm.py: generate a weighted wrinkle map from a groundtruth
- train_wrinkle_wds.py: train a wrinkle segmentation model using weighted deep supervision
- train_retinal_agnet.py: train a retinal vessel segmentation model using weighted deep supervision
- train_retinal_agnet_aspp.py: train a retinal vessel segmentation model using weighted deep supervision and ASPP
- inference_wrinkle.py: inference wrinkles from face images
- inference_retinal_vessel.py: inference vessels from  retinal images

Some of the code is copied without the developer's consent and may be deleted later. However, you can download it from the original author's repository.
- U-Net Pytorch: https://github.com/milesial/Pytorch-UNet
- AGNet: https://github.com/HzFu/AGNet
