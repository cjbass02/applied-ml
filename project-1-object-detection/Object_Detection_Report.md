# Object Detection with YOLOv3 — Report

This report summarizes results from running a YOLOv3 detector (pre-trained on COCO)
on the provided images and two additional web images. The notebook outputs are saved
in `outputs/` and referenced below as figures.

## 1) Included Images: Detection Performance

Overall, YOLOv3 detected the main objects in each image with high confidence. The
bounding boxes were generally well aligned, though some smaller objects and edge
cases were less precise.

- `IMG_1703.jpeg` (classroom or lecture scene): Detected multiple people and chairs,
  plus a laptop. The people and large chairs are well localized; some chairs are
  over-segmented or slightly mis-sized.  
  Figure: `outputs/IMG_1703_detected.jpeg`
- `IMG_1704.jpeg` (small group with chairs): Detected several people and multiple
  chairs. Localization is decent despite small subject size. Some persons appear
  fragmented into multiple overlapping boxes.  
  Figure: `outputs/IMG_1704_detected.jpeg`
- `IMG_4134.jpeg` (dog): Correctly detected the dog. Several small “book” detections
  appear near the top edge, likely false positives. The dog box is accurate.  
  Figure: `outputs/IMG_4134_detected.jpeg`
- `IMG_5415.jpg` (cars): Detected a large car with high confidence and a second car
  near the left edge. The large car box is good; the edge car is partially visible,
  so the box is less stable.  
  Figure: `outputs/IMG_5415_detected.jpg`
- `IMG_7207.jpg` (two people): Detected two people with reasonable bounding boxes,
  though one box is a bit tight.  
  Figure: `outputs/IMG_7207_detected.jpg`

## 2) Own Images (Web): Detection Performance

Two sample images from the web were evaluated to satisfy the “own images” requirement.

- `web_dog.jpg`: Correctly detected the dog, bicycle, and truck. The bike and truck
  are accurate; the dog box is tight around the body.  
  Figure: `outputs/web_dog_detected.jpg`
- `web_giraffe.jpg`: Correctly detected the giraffe and zebra. Both boxes are
  well localized; the zebra is partially occluded but still detected.  
  Figure: `outputs/web_giraffe_detected.jpg`

## 3) Competing Models and How YOLO Differs

**Competing models:**  
Two‑stage detectors such as R‑CNN, Fast R‑CNN, and Faster R‑CNN use a region proposal
stage followed by classification and box refinement. One‑stage detectors such as SSD
and RetinaNet predict boxes and class probabilities directly from feature maps.

**How YOLO differs:**  
YOLO is a single‑stage detector that frames detection as a direct regression problem.
It predicts bounding boxes and class probabilities in one forward pass, which makes
it fast and suitable for real‑time use. This speed can trade off with accuracy,
especially for small objects, but later versions (YOLOv3) improve performance with
multi‑scale predictions and stronger feature extraction.

## 4) Ethical Issues

### (a) Privacy risks in large datasets and trained models

- **Dataset privacy:** Large image datasets can include identifiable faces, license
  plates, or other sensitive information, often scraped without consent.
- **Model privacy:** Even if the dataset is not public, the trained model can leak
  information through membership inference or model inversion attacks. Sharing the
  model can therefore expose sensitive attributes or examples the model memorized.

### (b) Criteria to check for bias and strategies to reduce it

**Criteria for bias audits:**
- Representation balance across demographic groups (race, gender, age, skin tone).
- Per‑group error rates (false positives/negatives) and confidence calibration.
- Performance across different lighting, backgrounds, and contexts that correlate
  with protected attributes.
- Label taxonomy checks for offensive or harmful class names and annotations.

**Strategies to mitigate bias:**
- Curate balanced datasets with targeted data collection for under‑represented groups.
- Apply reweighting or stratified sampling during training.
- Use bias‑aware evaluation metrics and publish disaggregated results.
- Incorporate human review for sensitive categories and remove problematic labels.
- Apply post‑processing or threshold adjustments to equalize error rates where
  appropriate.

## Figures

Included images:
- `outputs/IMG_1703_detected.jpeg`
- `outputs/IMG_1704_detected.jpeg`
- `outputs/IMG_4134_detected.jpeg`
- `outputs/IMG_5415_detected.jpg`
- `outputs/IMG_7207_detected.jpg`

Own images (web):
- `outputs/web_dog_detected.jpg`
- `outputs/web_giraffe_detected.jpg`
