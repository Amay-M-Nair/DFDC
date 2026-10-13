---
artifact_type: compression_artifacts
regions: background / face boundary, left cheek, right cheek
citation: Rössler et al., "FaceForensics++: Learning to Detect Manipulated Facial Images", ICCV 2019
---
FaceForensics++ showed that detection accuracy drops sharply under heavy compression, because compression destroys high-frequency manipulation traces. Compression also adds its own block artefacts, which can be mistaken for manipulation. A manipulated region that was compressed twice (once before and once after editing) may show a JPEG grid or quantisation pattern that differs from the rest of the image. Attention aligned with 8x8 blocking or on flat areas of a heavily compressed image should be treated cautiously, because it may reflect compression rather than forgery.
