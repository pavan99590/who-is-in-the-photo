# Research and Design Rationale

This note records the sources and trade-offs relevant to the implemented design. The code and the project's own evaluation files are the evidence for what this application actually does; results from other models or datasets should not be read as this app's accuracy.

## Requirements that shaped the research

The assessment calls for identifying people in a group photo against a user-selected folder, marking people not in the folder as **Unknown**, and working on one laptop without online APIs or internet at runtime. That is an **open-set identification** task: the system compares each detected face with a gallery, but must be allowed to reject every gallery match.

This ruled out hosted recognition APIs as implementation choices, regardless of convenience. It also made local model availability, CPU inference, safe rejection of uncertain matches, and a straightforward non-technical UI important design criteria.

## Options compared

| Option | What it offers | Decision for this project |
|---|---|---|
| Hosted face-recognition API | A service can provide face-matching endpoints without bundling local models. | Rejected: sending photos to a hosted service conflicts with the explicit offline/no-online-API rule. |
| `face_recognition` with dlib | A well-known local Python face-recognition library with a simple API. | Considered as a local alternative. This project instead uses InsightFace, whose `FaceAnalysis` interface supplies the detector/recognition pipeline and whose model packs are available as ONNX files. This is a fit decision, not a claim that one library is universally more accurate. |
| InsightFace with local ONNX models | A local face-analysis library with detector and recognition models, compatible with CPU execution. | Selected. The application explicitly configures `CPUExecutionProvider`, loads the `buffalo_l` pack from its local model directory, and refuses to fetch missing model files during app runtime. |
| Nearest-gallery match with no rejection | Always returns the closest person in the folder. | Rejected: a stranger would always receive a person's name. The implementation uses a threshold and returns `Unknown` when the match is not accepted. |

InsightFace's model-zoo documentation describes the `buffalo_l` detection/recognition pack and states that its pretrained models are for **non-commercial research purposes only**. The project code is separately licensed; check the model terms before any use beyond the assessment. See [InsightFace model zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo) and [InsightFace model usage and licensing](https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md).

## Recognition approach

InsightFace produces a face embedding for each detected face. The project compares normalized embeddings using cosine similarity, then applies a one-to-one assignment so one gallery identity cannot be assigned to multiple detected faces in a single photo. A configurable similarity threshold determines whether the assigned identity is shown or the face is labeled **Unknown**. These decisions are implemented in [faceid/engine.py](./faceid/engine.py), [faceid/pipeline.py](./faceid/pipeline.py), and [faceid/matcher.py](./faceid/matcher.py).

The model's ArcFace recognition lineage is described in the original [ArcFace CVPR 2019 paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html). The paper motivates discriminative face embeddings; it does not establish accuracy for this specific application or its threshold.

The implementation separates detection/embedding from matching. This lets the UI re-apply a new threshold to saved detections without repeating the slower face-detection pass. This is an application-level responsiveness choice, not a model change.

## Data and evaluation

The project uses the **Labeled Faces in the Wild (LFW)** dataset to build a repeatable, small identity test set. The test-set script selects 50 gallery identities with at least two images, uses a different image of each for 50 known probes, and samples up to 50 other identities as unknown probes with seed 42. See the [scikit-learn LFW dataset documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html), [scripts/make_lfw_testset.py](./scripts/make_lfw_testset.py), and [scripts/evaluate_probes.py](./scripts/evaluate_probes.py).

The repository reports these observed checks at threshold 0.40:

- **LFW identity probes:** 49 of 50 known probes were named correctly; no wrong names or strangers named.
- **Synthetic group collages:** 26 of 27 present gallery people were named correctly; no wrong names or strangers named; the no-face image produced no detections.
- **One real group photo:** 35 faces were detected; one gallery person present was named and the other 34 faces stayed Unknown.

The group-photo results and their test conditions are in [docs/evidence/group_test_results.md](./docs/evidence/group_test_results.md). These are small, specific test sets—not an accuracy estimate or a guarantee for unseen office photos. LFW's distribution and the single real-photo case do not represent every organization or demographic. The [NIST Face Recognition Technology Evaluation: Demographic Effects](https://pages.nist.gov/frvt/html/frvt_demographics.html) documents demographic variation across face-recognition algorithms and evaluations; that is a reason to test and interpret local results carefully, not a measurement of this project's demographic performance.

The project does not include the LFW image collection in its test-data creation instructions; it asks the developer to fetch the data when rebuilding the test set. Follow the source dataset and model licenses and do not treat a public download as blanket permission for every use.

## Why these choices fit the demo—and their limits

- **Local CPU inference:** satisfies the offline/laptop constraint and avoids a hosted inference dependency. It requires a one-time dependency/model download during setup and is slower than a suitably provisioned GPU.
- **Thresholded open-set matching:** can leave an uncertain or out-of-gallery face Unknown instead of forcing a name. The threshold is empirical and can trade missed identities against false names; it should not be treated as universally calibrated.
- **One reference photo per person:** keeps folder setup simple and follows the assessment format. It gives less appearance coverage than a multi-photo enrollment system.
- **One-to-one assignment:** prevents duplicate identity assignments in one group photo. Consequently, if the same person appears more than once, only one occurrence can receive that identity.
- **Small, transparent UI:** a local browser UI avoids a separate hosted frontend and makes folder selection, photo upload, the marked result, and the results list available in one workflow.

## Sources

1. [Industry.AI take-home assessment](https://github.com/pavan99590/who-is-in-the-photo) — project context and task constraints.
2. [InsightFace Model Zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo) — model-pack overview and model-use notice.
3. [InsightFace model usage and licensing](https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md) — ONNX model usage and license distinction.
4. [ArcFace: Additive Angular Margin Loss for Deep Face Recognition (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html) — embedding method research context.
5. [Labeled Faces in the Wild dataset documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html) — benchmark description and dataset loader.
6. [NIST FRTE: Demographic Effects in Face Recognition](https://pages.nist.gov/frvt/html/frvt_demographics.html) — independent evaluation context for demographic variation.
7. [`face_recognition` project](https://github.com/ageitgey/face_recognition) — example of an alternative local Python face-recognition library.
8. [Project evaluation evidence](./docs/evidence/group_test_results.md), [LFW test-set builder](./scripts/make_lfw_testset.py), and [probe evaluator](./scripts/evaluate_probes.py) — this project's actual test protocol and reported results.
