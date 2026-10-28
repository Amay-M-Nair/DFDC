# Blending boundary

## bb-face-xray
artifact: blending_boundary
kind: artifact
regions: chin, left jawline, right jawline, left cheek, right cheek, hairline, face boundary
pattern: outer
families: swap_autoencoder, swap_graphics, swap_gan, reenact_graphics, reenact_neural
source: Li et al., "Face X-Ray for More General Face Forgery Detection", CVPR 2020
checked: no

Most face manipulation methods create an altered face and then blend it into the target image. Face X-Ray detects the resulting blending boundary, a band around the face outline where the two sources meet, which lets it generalise to manipulation methods it was not trained on.

## bb-colour-mask
artifact: blending_boundary
kind: artifact
regions: left cheek, right cheek, chin, left jawline, right jawline, forehead, hairline, face boundary
pattern: outer
families: swap_autoencoder
source: Li et al., "Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics", CVPR 2020
checked: no

Li et al. list colour mismatch between the synthesised and the original face, and visible boundaries of inaccurate face masks, as common artefacts of earlier DeepFake videos. Celeb-DF was made with colour correction and improved masks to reduce them.

## bb-scope
artifact: blending_boundary
kind: scope
regions: nose, mouth, teeth, left eye, right eye
pattern: central
families: any
source: Li et al., "Face X-Ray for More General Face Forgery Detection", CVPR 2020
checked: no

A blending boundary lies where the inserted face meets the rest of the image: the face outline, jawline and hairline. Evidence concentrated on inner features such as the nose or mouth does not by itself point to a blending boundary.
