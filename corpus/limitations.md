# Limitations of the evidence
What the detector and its heatmap cannot show. One of these cards is added to every grounded prompt.
A card with a `confidence` line is preferred for those confidence levels.

## lim-where-not-why
artifact: -
kind: limitation
regions: all
pattern: any
families: any
confidence: high
source: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017
checked: no

Grad-CAM shows where in the image the evidence for the model's decision lies, not what that evidence is. The highlighted region can be named, but the type of artefact the model reacted to cannot be read from the heatmap itself.

## lim-coarse-map
artifact: -
kind: limitation
regions: all
pattern: any
families: any
source: Selvaraju et al., "Grad-CAM", ICCV 2017; Chattopadhay et al., "Grad-CAM++: Generalized Gradient-Based Visual Explanations for Deep Convolutional Networks", WACV 2018
checked: no

Grad-CAM and Grad-CAM++ are computed on the last convolutional layer, whose feature map is much coarser than the image, and then upsampled. They localise areas such as the centre of the face, not fine structures such as single teeth or a thin blending seam.

## lim-sanity-checks
artifact: -
kind: limitation
regions: all
pattern: any
families: any
source: Adebayo et al., "Sanity Checks for Saliency Maps", NeurIPS 2018
checked: no

Adebayo et al. showed that some saliency methods produce similar maps even when the model's weights are randomised, so a plausible-looking map is not proof of what the model used. Saliency maps need to be tested before they are trusted.

## lim-binary-detector
artifact: -
kind: limitation
regions: all
pattern: any
families: any
source: This project, phase 1: binary real/fake classifier trained on FaceForensics++
checked: no

The detector only outputs real or fake. It was not trained to recognise the manipulation method, so the method or generator behind a fake cannot be inferred from its output.

## lim-single-frame
artifact: -
kind: limitation
regions: all
pattern: any
families: any
source: Li, Chang & Lyu, "In Ictu Oculi: Exposing AI Created Fake Videos by Detecting Eye Blinking", WIFS 2018; Li et al., "Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics", CVPR 2020
checked: no

Some deepfake cues are temporal, such as unnatural eye blinking or flickering between frames. A single frame, as analysed here, cannot show them.

## lim-generalisation
artifact: -
kind: limitation
regions: all
pattern: any
families: any
confidence: moderate, borderline
source: Li et al., "Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics", CVPR 2020
checked: no

Detectors trained on one dataset lose much of their accuracy on deepfakes made with other methods: Li et al. report large drops on Celeb-DF for detectors that perform well on earlier datasets. A verdict near the decision threshold is therefore uncertain.
