# Group-photo test results

Date: 22 Nov 2026 | Threshold: 0.40 | Detector: SCRFD (InsightFace buffalo_l), embeddings: ArcFace, CPU only

## A. Synthetic collages (testset_collage, seed 7)
Built from the LFW test set: 5 photos of 10 faces each (a mix of gallery people, shown in a different photo from their reference, and strangers), plus one photo with no faces.

- Faces: 53 found vs 50 expected. The 3 extra are small background faces inside LFW tiles, and all stayed Unknown.
- People present in the gallery: 27. Named correctly: 26. Wrong names: 0. Missed: 1 (cause not yet checked).
- Strangers given a name: 0.
- Photo with no faces: 0 faces detected.

## B. Real photo (1 group photo, 35 faces)
Gallery: 18 people (3 real, 15 LFW decoys). Present in the photo: 1 (Person A).

- Person A: named correctly, similarity 0.80, margin +0.62 over the next person.
- The 2 other real people in the gallery were not in the photo, and no face was matched to them.
- The other 34 faces: all Unknown.
- Faces detected: 35 of 35. Time: about 8 seconds on CPU.

Result: 1 of 1 present person named, 0 wrong names among 35 faces. This is one photo with one positive case. It shows the pipeline works on a real crowded photo. It is not an accuracy estimate.

## C. Edge cases
| Case | Expected | Observed |
|---|---|---|
| Photo with no faces | "No faces found", no crash | (to fill in) |
| Folder with no photos | clear message, no crash | (to fill in) |
| File that is not a photo | clear message, no crash | (to fill in) |

## Limits
- LFW is mostly public figures, many of them Western. Recognition accuracy and false-match rates can differ across groups of people, so a threshold tuned on LFW may not transfer to other people.
- Collage faces are separate frontal tiles on a plain background. They do not test overlapping faces, shared lighting, hats, glasses or head angles.
- The real-photo test is a single photo. Counts are reported as counts, not percentages.