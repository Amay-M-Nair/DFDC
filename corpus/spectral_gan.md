# GAN spectral fingerprints

## gan-upconvolution
artifact: spectral_gan_fingerprint
kind: artifact
regions: forehead / hairline, left cheek, right cheek, background / face boundary
pattern: outer
families: synthetic
source: Durall, Keuper & Keuper, "Watch Your Up-Convolution: CNN Based Generative Deep Neural Networks Are Failing to Reproduce Spectral Distributions", CVPR 2020
checked: no

Durall et al. showed that the up-sampling layers of generative networks fail to reproduce the high-frequency spectrum of natural images. The resulting spectral distortions can be detected even when they are invisible to people.

## gan-frequency-grid
artifact: spectral_gan_fingerprint
kind: artifact
regions: forehead / hairline, left cheek, right cheek, background / face boundary
pattern: outer
families: synthetic
source: Frank et al., "Leveraging Frequency Analysis for Deep Fake Image Recognition", ICML 2020
checked: no

Frank et al. found grid-like patterns in the frequency spectrum of GAN-generated images, caused by up-sampling, which make such images easy to separate from real ones in the frequency domain.

## gan-scope
artifact: spectral_gan_fingerprint
kind: scope
regions: all
pattern: any
families: any
source: Rössler et al., "FaceForensics++", ICCV 2019; Li et al., "Celeb-DF", CVPR 2020
checked: no

Spectral fingerprints were shown for fully synthetic images, in which a GAN generates every pixel. Face swaps and reenactments, such as those in FaceForensics++ and Celeb-DF, alter only the face and blend it into a real video frame, so a GAN-generated image cannot be assumed for them.
