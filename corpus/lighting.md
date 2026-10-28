# Lighting inconsistencies

## light-estimation
artifact: lighting_inconsistency
kind: artifact
regions: nose, left cheek, right cheek, forehead
pattern: central
families: swap_autoencoder, swap_graphics, reenact_graphics
source: Matern, Riess & Stamminger, "Exploiting Visual Artifacts to Expose Deepfakes and Face Manipulations", WACV Workshops 2019
checked: no

Matern et al. point out that manipulations which transfer or re-render a face depend on an estimate of the scene's lighting; errors leave shading or highlights on the face that do not match the rest of the scene.

## light-face2face
artifact: lighting_inconsistency
kind: artifact
regions: nose, left cheek, right cheek, forehead
pattern: central
families: reenact_graphics
source: Thies et al., "Face2Face: Real-time Face Capture and Reenactment of RGB Videos", CVPR 2016
checked: no

Face2Face re-renders the target's face with a parametric face model under an estimated scene illumination. Where the estimated lighting differs from the real one, the re-rendered face can show shading that does not match the scene.
