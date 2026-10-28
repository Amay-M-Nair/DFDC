# Hair and background artefacts

## hb-inner-face
artifact: background_hair_artifacts
kind: artifact
regions: hairline, face boundary
pattern: outer
families: swap_autoencoder, swap_graphics, swap_gan
source: Rössler et al., "FaceForensics++: Learning to Detect Manipulated Facial Images", ICCV 2019
checked: no

Face-swap methods replace only the inner face region, while hair, ears and background come from the target video. Mismatches can therefore appear where the swapped face meets the hairline and the surrounding image.

## hb-whole-image
artifact: background_hair_artifacts
kind: artifact
regions: hairline, face boundary, background
pattern: outer
families: synthetic
source: Karras et al., "Analyzing and Improving the Image Quality of StyleGAN", CVPR 2020
checked: no

A fully synthetic face is generated as a whole image, so artefacts can appear outside the face as well, such as the blob-shaped artefacts that Karras et al. found in images from the original StyleGAN.

## hb-scope
artifact: background_hair_artifacts
kind: scope
regions: hairline, face boundary, background
pattern: outer
families: any
source: Rössler et al., "FaceForensics++: Learning to Detect Manipulated Facial Images", ICCV 2019
checked: no

In face-swap and reenactment videos the background is taken unchanged from the original video, so evidence on the background alone is weak support for manipulation, except along the face boundary.
