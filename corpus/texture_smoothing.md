# Texture smoothing

## tex-global-texture
artifact: texture_smoothing
kind: artifact
regions: left cheek, right cheek, forehead / hairline, chin / jawline
pattern: outer
families: synthetic
source: Liu et al., "Global Texture Enhancement for Fake Face Detection in the Wild", CVPR 2020
checked: no

Liu et al. found that the skin texture of GAN-generated faces differs from real skin, and that detectors using global texture statistics are more robust to image distortions.

## tex-low-resolution
artifact: texture_smoothing
kind: artifact
regions: left cheek, right cheek, forehead / hairline, chin / jawline
pattern: outer
families: swap_autoencoder
source: Li & Lyu, "Exposing DeepFake Videos By Detecting Face Warping Artifacts", CVPR Workshops 2019; Li et al., "Celeb-DF", CVPR 2020
checked: no

A face synthesised at low resolution and enlarged to fit the target looks smoother and blurrier than the surrounding image. Li et al. name low-resolution synthesised faces as a visible artefact of earlier DeepFake datasets.
