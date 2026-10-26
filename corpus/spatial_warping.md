# Spatial warping and geometry

## warp-affine
artifact: spatial_warping
kind: artifact
regions: nose, mouth / teeth, left cheek, right cheek, chin / jawline
pattern: central
families: swap_autoencoder
source: Li & Lyu, "Exposing DeepFake Videos By Detecting Face Warping Artifacts", CVPR Workshops 2019
checked: no

DeepFake methods synthesise the face at a limited resolution and then warp it to match the pose and shape of the target face. Li and Lyu detect the resulting resolution inconsistency between the warped face area and its surroundings.

## warp-head-pose
artifact: spatial_warping
kind: artifact
regions: nose, mouth / teeth, chin / jawline
pattern: central
families: swap_autoencoder
source: Yang, Li & Lyu, "Exposing Deep Fakes Using Inconsistent Head Poses", ICASSP 2019
checked: no

Yang et al. found that splicing a synthesised central face into an original image makes the head pose estimated from the central features (such as the nose and mouth) disagree with the pose estimated from the whole face.

## warp-3d-fit
artifact: spatial_warping
kind: artifact
regions: nose, mouth / teeth, left cheek, right cheek, chin / jawline
pattern: central
families: swap_graphics
source: Rössler et al., "FaceForensics++: Learning to Detect Manipulated Facial Images", ICCV 2019
checked: no

FaceSwap, a graphics-based method, fits a 3D face template to sparse facial landmarks and renders it onto the target face before blending and colour correction. A face shape fitted from few landmarks can distort the face's geometry.

## warp-scope
artifact: spatial_warping
kind: scope
regions: nose, mouth / teeth, left cheek, right cheek, chin / jawline
pattern: central
families: any
source: Li et al., "Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics", CVPR 2020
checked: no

Celeb-DF synthesises faces at a higher resolution than earlier DeepFake datasets to reduce visible low-resolution artefacts, so warping traces are weaker in more recent face swaps.
