# Compression artefacts

## comp-block-traces
artifact: compression_artifacts
kind: artifact
regions: background / face boundary, left cheek, right cheek
pattern: outer
families: any
source: Wallace, "The JPEG Still Picture Compression Standard", IEEE Trans. Consumer Electronics 1992; Bianchi & Piva, "Image Forgery Localization via Block-Grained Analysis of JPEG Artifacts", IEEE TIFS 2012
checked: no

Block-based codecs such as JPEG compress an image in 8x8 blocks, which can leave blocking artefacts in smooth areas. When a region is pasted into a compressed image, its compression traces can differ from those of the rest of the image, which forensic methods use to localise tampering.

## comp-scope
artifact: compression_artifacts
kind: scope
regions: all
pattern: any
families: any
source: Rössler et al., "FaceForensics++: Learning to Detect Manipulated Facial Images", ICCV 2019
checked: no

In FaceForensics++, real and manipulated videos are released at the same compression levels, and detection accuracy drops under strong compression because it removes fine manipulation traces. Compression artefacts alone do not show that a face was manipulated.
