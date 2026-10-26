# Authentic cues (artifact: none)
Cards for REAL verdicts: what the detector's evidence means when it judges a face authentic.
Card format: see src/explain/corpus.py.

## auth-class-evidence
artifact: none
kind: authentic
regions: all
pattern: any
families: any
source: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017
checked: no

Grad-CAM highlights the image regions that support the class the model predicted. When the verdict is real, the highlighted regions are where the model found evidence of an authentic face, not the location of an artefact.

## auth-skin-texture
artifact: none
kind: authentic
regions: left cheek, right cheek, forehead / hairline, chin / jawline
pattern: outer
families: any
source: Liu et al., "Global Texture Enhancement for Fake Face Detection in the Wild", CVPR 2020
checked: no

Liu et al. found that the skin texture of real faces differs from that of generated faces, and that global texture statistics help detectors tell them apart. Evidence on large skin areas of a face judged real is consistent with the detector relying on natural skin texture.

## auth-eye-highlights
artifact: none
kind: authentic
regions: left eye, right eye
pattern: central
families: any
source: Hu, Li & Lyu, "Exposing GAN-Generated Faces Using Inconsistent Corneal Specular Highlights", ICASSP 2021
checked: no

In a real photograph both eyes see the same scene, so the reflections of light sources on the two corneas have similar shapes and positions. Consistent highlights in both eyes are a cue of an authentic face.

## auth-mouth-detail
artifact: none
kind: authentic
regions: mouth / teeth
pattern: central
families: any
source: Matern, Riess & Stamminger, "Exploiting Visual Artifacts to Expose Deepfakes and Face Manipulations", WACV Workshops 2019
checked: no

Matern et al. note that manipulated faces often render teeth without individual detail. Evidence on the mouth of a face judged real is consistent with the detector relying on natural detail there, such as separate teeth.
