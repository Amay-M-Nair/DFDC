---
artifact_type: blending_boundary
regions: chin / jawline, left cheek, right cheek, forehead / hairline, background / face boundary
citation: Li et al., "Face X-Ray for More General Face Forgery Detection", CVPR 2020
---
Most face-swap pipelines generate or transfer an inner face and then composite it onto a target image. The compositing step leaves a blending boundary: a thin band around the face outline where the two sources meet. Face X-Ray models forgery detection as finding this boundary instead of learning a particular generator's artefacts, which helps it generalise to unseen manipulation methods. Visible cues include colour or skin-tone discontinuities along the jawline, cheeks or hairline, a softened or doubled contour at the face edge, and noise statistics that differ inside and outside the face mask. Detector attention that concentrates on the face perimeter rather than on internal features is consistent with a blending-boundary artefact.
