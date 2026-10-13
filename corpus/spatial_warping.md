---
artifact_type: spatial_warping
regions: left cheek, right cheek, chin / jawline, nose, mouth / teeth
citation: Li & Lyu, "Exposing DeepFake Videos By Detecting Face Warping Artifacts", CVPR Workshops 2019
---
Early deepfake generators synthesise faces at a fixed low resolution, then apply affine warping to fit the target's pose and face shape. Resampling and warping leave resolution inconsistencies: the warped face area is blurrier or has different sharpness from its surroundings, and facial geometry can look subtly distorted, for example asymmetric cheeks, a bent nose bridge or a stretched mouth. Detectors that focus on the cheeks, nose or jaw, where warping is strongest, may be responding to these resampling artefacts.
