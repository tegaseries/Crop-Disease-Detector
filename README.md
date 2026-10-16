# Crop Disease Detector

Upload a photo of a single leaf and the app names the crop and its condition (healthy or a specific disease). The model is MobileNetV2, adapted by transfer learning and fine-tuning to 39 classes: 38 healthy or diseased conditions across 14 crops, plus a background class for photos with no leaf in them.

Capstone project, TechCrush AI/ML Bootcamp, Cohort 8 (Class A), October 2026.

**Live app:** https://crop-disease-detector-techcrush.streamlit.app (Streamlit Community Cloud)

## Results

Measured on 6,168 test images that played no part in training:

| Metric | Value |
|---|---|
| Test accuracy | **98.43%** |
| Macro-average precision / recall / F1 | 0.981 / 0.981 / 0.980 |
| Weighted-average F1 | 0.984 |
| Misclassified test images | 97 of 6,168 |

- 24 of the 39 classes reached an F1-score of 0.99 or higher, and 8 had no errors at all.
- The weakest class was potato late blight (F1 0.922). Every potato late blight leaf was recognised, but 17 images from other classes were wrongly labelled as it, including 5 tomato late blight leaves (both diseases are caused by the same pathogen).
- The most common single mistake was tomato spider-mite damage taken for target spot (11 times).
- Only 4 diseased leaves (0.06% of the test set) were passed as healthy.

**Read 98.43% as an upper estimate.** The dataset's authors added augmented (rotated, flipped) copies of their images before publishing it, and the split was made afterwards, so copies of the same leaf can sit in both the training and test sets. All images are single leaves on plain backgrounds; photos taken in a field will score lower.

## How it works

1. **Data.** 61,486 leaf images in 39 classes from the Mendeley dataset *Plant leaf diseases dataset with augmentation* (built from PlantVillage, plus a background class).
2. **Split.** 80:10:10 with `split-folders` (seed 1337): 49,179 training, 6,139 validation and 6,168 test images.
3. **Preprocessing.** Every image is resized to 160 × 160 pixels. MobileNetV2's rescaling to −1…1 is built into the model, so the app only has to resize.
4. **Phase 1, feature extraction.** The pre-trained MobileNetV2 base (ImageNet weights) is frozen and only a new 39-class output layer is trained: 49,959 trainable parameters, 10 epochs.
5. **Phase 2, fine-tuning.** Layers 100 to 153 of the base are unfrozen (1,911,399 trainable parameters) and trained for 20 more epochs.
6. **Evaluation.** Per-class precision, recall and F1-score and a confusion matrix, as well as overall accuracy.

| Layer | Output shape | Parameters |
|---|---|---|
| Input | 160 × 160 × 3 | 0 |
| Rescaling to −1…1 | 160 × 160 × 3 | 0 |
| MobileNetV2 base | 5 × 5 × 1,280 | 2,257,984 |
| Global average pooling | 1,280 | 0 |
| Dropout (0.2) | 1,280 | 0 |
| Dense, sigmoid | 39 | 49,959 |
| **Total** | | **2,307,943** |

Training ran on a free Google Colab T4 GPU in about 35 minutes, with the Adam optimiser and sparse categorical cross-entropy.

## Run it on your own computer

You need Python 3.12 or 3.13.

```
pip install -r requirements.txt
streamlit run main.py
```

Your browser then opens the app. Keep `main.py` and `crop_disease_detector_model.keras` in the same folder, because the app loads the model by its file name.

## What's in this repository

| File | What it is |
|---|---|
| `main.py` | The Streamlit app |
| `crop_disease_detector_model.keras` | The trained model (24 MB) |
| `requirements.txt` | The packages the app needs, at the versions it was tested with |
| `Crop_Disease_Detector_Machine_Learning.ipynb` | The training notebook (Google Colab), with its saved outputs |

## Versions

- **1.1:** every upload is converted to RGB, so PNG images with a transparent background and black-and-white photos now work (in 1.0 they stopped the app with an error). Only JPG and PNG files can be uploaded. The Home page text was rewritten, and the Predict button is now called Diagnose.
- **1.0:** first version.

## Known issues (version 1.1)

- The app shows only the predicted class, without a confidence score, so an unsure guess looks the same as a sure one.
- The model is loaded again on every click, which makes each diagnosis take several seconds.

## Crops and conditions covered

<details>
<summary>All 39 classes, in the model's output order</summary>

| # | Crop | Condition |
|---|---|---|
| 0 | Apple | Apple scab |
| 1 | Apple | Black rot |
| 2 | Apple | Cedar apple rust |
| 3 | Apple | Healthy |
| 4 | (none) | Background, no leaf |
| 5 | Blueberry | Healthy |
| 6 | Cherry | Powdery mildew |
| 7 | Cherry | Healthy |
| 8 | Maize (corn) | Cercospora leaf spot (grey leaf spot) |
| 9 | Maize (corn) | Common rust |
| 10 | Maize (corn) | Northern leaf blight |
| 11 | Maize (corn) | Healthy |
| 12 | Grape | Black rot |
| 13 | Grape | Esca (black measles) |
| 14 | Grape | Leaf blight (Isariopsis leaf spot) |
| 15 | Grape | Healthy |
| 16 | Orange | Huanglongbing (citrus greening) |
| 17 | Peach | Bacterial spot |
| 18 | Peach | Healthy |
| 19 | Bell pepper | Bacterial spot |
| 20 | Bell pepper | Healthy |
| 21 | Potato | Early blight |
| 22 | Potato | Late blight |
| 23 | Potato | Healthy |
| 24 | Raspberry | Healthy |
| 25 | Soybean | Healthy |
| 26 | Squash | Powdery mildew |
| 27 | Strawberry | Leaf scorch |
| 28 | Strawberry | Healthy |
| 29 | Tomato | Bacterial spot |
| 30 | Tomato | Early blight |
| 31 | Tomato | Late blight |
| 32 | Tomato | Leaf mould |
| 33 | Tomato | Septoria leaf spot |
| 34 | Tomato | Spider mites (two-spotted spider mite) |
| 35 | Tomato | Target spot |
| 36 | Tomato | Yellow leaf curl virus |
| 37 | Tomato | Mosaic virus |
| 38 | Tomato | Healthy |

</details>

## Team

- Aghe Promise Oghenetega ([@tegaseries](https://github.com/tegaseries))
- Nnaemeka Donatus Nwose ([@nnaemekanwose](https://github.com/nnaemekanwose))
- Otuokpai Happiness Joy ([@otuokpaihappinessjoy-commits](https://github.com/otuokpaihappinessjoy-commits))
- Anyalechi Angel Chinwemmeri ([@angelanyalechi](https://github.com/angelanyalechi))

## Acknowledgements

- Dataset: J. Arun Pandian and G. Geetharamani, "Data for: Identification of plant leaf diseases using a 9-layer deep convolutional neural network," Mendeley Data, V1, 2019, doi: [10.17632/tywbtsjrjv.1](https://data.mendeley.com/datasets/tywbtsjrjv/1).
- Model: M. Sandler et al., "MobileNetV2: Inverted residuals and linear bottlenecks," CVPR 2018.
- Method: the TensorFlow tutorial [Transfer learning and fine-tuning](https://www.tensorflow.org/tutorials/images/transfer_learning).
- Thanks to the TechCrush AI/ML Bootcamp facilitators.

## License

No license has been chosen yet; the team will decide. Until then, the code can be viewed but not reused.
