# Eye inconsistencies

## eye-highlights
artifact: eye_inconsistency
kind: artifact
regions: left eye, right eye
pattern: central
families: synthetic
source: Hu, Li & Lyu, "Exposing GAN-Generated Faces Using Inconsistent Corneal Specular Highlights", ICASSP 2021
checked: no

Hu et al. found that GAN-generated faces often have corneal specular highlights that differ between the two eyes in shape or position, whereas in real photographs both eyes reflect the same light sources.

## eye-colour-reflections
artifact: eye_inconsistency
kind: artifact
regions: left eye, right eye
pattern: central
families: synthetic, swap_autoencoder
source: Matern, Riess & Stamminger, "Exploiting Visual Artifacts to Expose Deepfakes and Face Manipulations", WACV Workshops 2019
checked: no

Matern et al. report irises of different colour in GAN-generated faces, and missing or simplified reflections in the eyes of deepfakes.

## eye-blink-scope
artifact: eye_inconsistency
kind: scope
regions: left eye, right eye
pattern: central
families: any
source: Li, Chang & Lyu, "In Ictu Oculi: Exposing AI Created Fake Videos by Detecting Eye Blinking", WIFS 2018
checked: no

Li et al. exposed early deepfake videos by their reduced eye blinking, which is a temporal cue. In a single frame, whether an eye is open or closed is not evidence of manipulation.
