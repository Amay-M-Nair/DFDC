# Teeth and mouth artefacts

## mouth-teeth-detail
artifact: teeth_mouth_artifacts
kind: artifact
regions: teeth
pattern: central
families: swap_autoencoder, reenact_graphics
source: Matern, Riess & Stamminger, "Exploiting Visual Artifacts to Expose Deepfakes and Face Manipulations", WACV Workshops 2019
checked: no

Matern et al. observed that manipulated faces often show the teeth as a single white area without individual teeth, and little detail inside the mouth.

## mouth-neuraltextures
artifact: teeth_mouth_artifacts
kind: artifact
regions: mouth, teeth, chin
pattern: central
families: reenact_neural
source: Rössler et al., "FaceForensics++", ICCV 2019; Thies, Zollhöfer & Nießner, "Deferred Neural Rendering: Image Synthesis using Neural Textures", ACM TOG 2019
checked: no

In FaceForensics++, the NeuralTextures manipulations change only the facial expression of the mouth region, which is re-rendered with neural rendering, so their traces concentrate around the mouth.

## mouth-face2face
artifact: teeth_mouth_artifacts
kind: artifact
regions: teeth
pattern: central
families: reenact_graphics
source: Thies et al., "Face2Face: Real-time Face Capture and Reenactment of RGB Videos", CVPR 2016
checked: no

Face2Face fills the mouth interior by retrieving a suitable mouth from other frames of the target video and warping it into place, so the teeth and inner mouth can fit the new expression poorly.
