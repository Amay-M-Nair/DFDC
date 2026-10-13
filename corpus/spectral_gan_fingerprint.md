---
artifact_type: spectral_gan_fingerprint
regions: forehead / hairline, left cheek, right cheek, background / face boundary, nose
citation: Durall, Keuper & Keuper, "Watch Your Up-Convolution: CNN Based Generative Deep Neural Networks Are Failing to Reproduce Spectral Distributions", CVPR 2020
---
The up-sampling layers in GANs (transposed convolutions, nearest or bilinear up-sampling followed by convolution) cannot reproduce the high-frequency spectrum of natural images. Fully synthetic images such as StyleGAN faces therefore carry a spectral fingerprint: periodic, checkerboard-like patterns and an abnormal high-frequency power distribution. These cues are often invisible to people but detectable by CNNs. They show up most clearly in large, smooth, low-texture areas such as the forehead, cheeks and plain backgrounds. Diffuse attention across smooth skin or background, without a single semantic focus, is consistent with a frequency-domain fingerprint.
