# cnn-transfer-learning
Intel Image Classification using CNN Transfer Learning with MobileNetV2. Feature extraction + fine-tuning, 93.5% accuracy on 6 classes (buildings, forest, glacier, mountain, sea, street). Kaggle notebook with augmentation, confusion matrix &amp; accuracy plots.
## Project 2, Week 5: Scene Classification with Transfer Learning

- **Dataset:** Intel Image Classification; 6 classes; 14,034 training images and 3,000 test images.
- **Model:** ResNet-18 pretrained on ImageNet.
- **Feature extraction:** 3,078 trainable parameters; best validation accuracy **90.1%**.
- **Fine-tuning:** Unfroze `layer4`, training 8,396,806 parameters; best validation accuracy **93.1%**.
- **Test accuracy (evaluated once):** **93.5%**.
- **Weakest classes by recall:** Glacier (87.0%), followed by buildings (90.2%).
- **Most-confused pair:** Glacier and mountain. According to the confusion-matrix analysis, 56 glacier images were predicted as mountain, while 30 mountain images were predicted as glacier.
- **Task 1.2 prediction:** Correctly anticipated that glacier and mountain images could be confused because of similar snow, ice, rocks, and terrain.
- **Other notable confusion:** Buildings and street, including 37 buildings images predicted as street according to the analysis.
- - **Next — Week 6:** Explore deep learning for skin-lesion image classification using a labeled medical image dataset, with the goal of distinguishing between different types of skin conditions.
- **Dataset link:** [HAM10000 — Skin Cancer MNIST](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
## Kaggle Notebook:https://www.kaggle.com/code/fatimaakhtar123/week1-cnn-intel-classification

## Week 6: Skin Lesion Classification for Skin Cancer Analysis

**Who benefits:** Researchers and students exploring deep learning for medical image classification in an educational or research setting.

**Why it matters:** Classifying skin lesions is a challenging image-classification problem because diagnostic categories are imbalanced and some lesions have similar visual appearances. Poor performance on less common classes can be hidden by overall accuracy.

**Dataset:** [HAM10000 — Skin Cancer MNIST](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000); license: CC BY-NC-SA 4.0, as listed on Kaggle. The dataset contains **10,015 images and 7 classes**. Diagnostic labels are provided in the dataset metadata. According to the dataset documentation, more than half of the lesions were confirmed through histopathology; the remaining diagnoses use follow-up examination, expert consensus, or in-vivo confocal microscopy.

**Data audit:** Class imbalance ratio: **58.30:1**. Exact duplicates: **4 files forming 2 identical pairs**, both confined to the training split. Test images with a near-duplicate in train below the 0.10 distance threshold: **0** out of 2,014 test images. Split: **6,380 train / 1,621 validation / 2,014 test** images, grouped by `lesion_id`, with **0 shared lesion groups** between any pair of splits. Patient-level separation could not be verified because patient IDs are unavailable.

**Model and imbalance handling:** Pretrained ResNet18 with an ImageNet initialization and a seven-class classification head. The training procedure used weighted cross-entropy loss, with balanced sampling disabled. Class weights: `akiec = 4.382`, `bcc = 2.770`, `bkl = 1.293`, `df = 12.152`, `mel = 1.334`, `nv = 0.212`, and `vasc = 11.252`.

**Results (held-out test set):** Accuracy: **0.745**; Macro-F1: **0.561**; weighted F1: **0.760**. Per-class recall: `akiec = 0.706`, `bcc = 0.730`, `bkl = 0.585`, `df = 0.227`, `mel = 0.563`, `nv = 0.811`, and `vasc = 0.706`. The weakest class by F1 was `df` at **0.286**, while `nv` achieved the highest F1 at **0.871**. Binary threshold selection was not applicable to the seven-class task.

**Grad-CAM:** Six test examples from the `bkl` class were inspected: three correct predictions and three misclassifications. Correct predictions generally showed attention around the lesion, while some incorrect predictions showed ring-shaped, diffuse, or separated attention regions. **Shortcut risk: suspected, not confirmed.** A proposed follow-up experiment is to mask or crop surrounding skin and compare predictions and performance with the original images.

**Limitations:** The dataset contains multi-source dermatoscopic images and has substantial class imbalance. The split prevents lesion-level leakage, but not necessarily patient-level leakage. Only six Grad-CAM examples were inspected, all from one class, and no independent external validation was performed.

**Not for:** Clinical diagnosis, treatment decisions, or unsupervised real-world skin cancer screening. This is an educational model and has not been clinically validated.
## Detailed Test Classification Report

The following table summarizes the model's performance on the held-out test set of 2,014 images.

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| `akiec` | 0.429 | 0.706 | 0.533 | 68 |
| `bcc` | 0.420 | 0.730 | 0.533 | 100 |
| `bkl` | 0.504 | 0.585 | 0.542 | 205 |
| `df` | 0.385 | 0.227 | 0.286 | 22 |
| `mel` | 0.483 | 0.563 | 0.520 | 222 |
| `nv` | 0.940 | 0.811 | 0.871 | 1,363 |
| `vasc` | 0.585 | 0.706 | 0.640 | 34 |
| **Accuracy** | — | — | **0.745** | **2,014** |
| **Macro average** | **0.535** | **0.618** | **0.561** | **2,014** |
| **Weighted average** | **0.790** | **0.745** | **0.760** | **2,014** |

**Metric interpretation:**
- **Precision:** How often predictions for a class were correct.
- **Recall:** The proportion of actual examples of a class correctly identified.
- **F1-score:** The harmonic mean of precision and recall.
- **Support:** The number of actual test examples belonging to each class.

**Key observation:** The `nv` class achieved the highest F1-score (0.871), while `df` had the lowest (0.286). The `df` recall of 0.227 indicates that the model missed many examples of this class. Overall accuracy alone therefore does not fully represent performance across all seven classes.

*These results are from an educational research model and do not establish clinical suitability for diagnosing skin lesions.*

### Responsible AI Reflection

**1. Who could be harmed, and how?**

Patients could be harmed if this model were used for real diagnosis or screening. It performs unevenly across classes: the recall for `df` was only **0.227**, and melanoma (`mel`) recall was **0.563**, meaning some lesions in these classes were missed. This could create false reassurance or delay medical assessment. Performance may also differ across skin tones, patient populations, and imaging conditions not adequately represented in the dataset.

**2. Where was the model not validated?**

The model was evaluated on a held-out test split from HAM10000, with lesions separated between training, validation, and test sets. However, it was **not externally validated** on an independent dataset, different hospitals, other imaging devices, diverse patient populations, or real clinical settings. Patient-level separation could not be confirmed because patient IDs were unavailable.

**3. What is the human role?**

A qualified dermatologist must remain responsible for clinical assessment and diagnosis. The model can only serve as an educational research tool, not as a replacement for medical expertise. Its predictions should not determine treatment, reassure patients that a lesion is harmless, or override professional judgment.

**4. What evidence would change your conclusions?**

I would evaluate the model on independent datasets from different clinical settings and populations, examine performance across all seven classes, and assess sensitivity, specificity, calibration, and clinically important errors. I would also test the suspected shortcut-learning risk by masking or cropping surrounding skin and comparing predictions against the original images. Consistent external performance, improved detection of underperforming classes, and evidence that predictions rely on relevant lesion features would strengthen confidence in the model—but clinical use would still require appropriate clinical validation.

# Week 7: Text and Multimodal AI

## Project Overview

This project explores Natural Language Processing (NLP), Named Entity Recognition (NER), Regular Expressions, Optical Character Recognition (OCR), and Multimodal AI.

The domain-specific application focuses on **Healthcare and Medical AI — Medical Imaging and Dermatology**.

## 1. Text Classification — SMS Spam Detection

**Dataset:** SMS Spam Collection Dataset (UCI)  
**Total messages:** 5,572

I trained two TF-IDF-based Logistic Regression classifiers using word and character n-grams.

| Model | Macro-F1 |
|---|---:|
| Word TF-IDF | 0.974 |
| Character n-grams | 0.982 |

Character n-grams performed slightly better than word n-grams.

**Top spam-associated features:** `call`, `txt`, `free`, `text`, `reply`, `uk`, `www`, `mobile`, `claim`, `stop`.

**Top ham-associated features:** `my`, `me`, `ok`, `that`, `it`, `but`, `ll`, `gt`, `lt`, `can`.

## 2. Named Entity Recognition (NER)

I evaluated spaCy's pre-trained NER model and improved it using Entity Ruler patterns.

### Original Pakistani Sample Sentences

| Metric | Before Entity Ruler | After Entity Ruler |
|---|---:|---:|
| Precision | 0.571 | 0.875 |
| Recall | 0.500 | 0.875 |
| F1-score | 0.533 | 0.875 |

### Healthcare and Medical AI Domain

The medical-domain evaluation used five development sentences and five held-out test sentences.

| Metric | Before Entity Ruler | After Entity Ruler |
|---|---:|---:|
| Precision | 0.500 | 0.500 |
| Recall | 0.500 | 0.500 |
| F1-score | 0.500 | 0.500 |

The medical-domain rules did not improve performance on the held-out sentences. Remaining challenges included recognizing disease names, dataset names, and complete monetary expressions.

## 3. Regular Expression Extraction

Developed domain-specific patterns for extracting:

- Patient IDs
- Medical record numbers
- Laboratory values
- ICD-10 diagnosis codes
- Medical imaging IDs
- CNIC numbers
- Mobile numbers
- Monetary amounts
- Dates
- Email addresses
- Invoice numbers

### Test Results

| Test category | Passed | Total |
|---|---:|---:|
| Positive tests | 12 | 12 |
| Negative tests | 8 | 8 |
| **Overall** | **20** | **20** |

All 20 tests passed. Invalid formats and unrelated information were correctly ignored.

## 4. OCR and Character Error Rate (CER)

Used EasyOCR to extract text from a synthetic invoice and applied context-based OCR correction rules.

| Evaluation | CER |
|---|---:|
| Raw OCR | 0.000 (0%) |
| After correction rules | 0.000 (0%) |

The synthetic invoice was recognized correctly without requiring additional corrections. Real phone photographs may produce higher CER due to blur, lighting, shadows, and perspective distortion.

## 5. Image + Text Multimodal Pipeline

Integrated my previously trained **ResNet18 HAM10000 skin-lesion classifier** with text evidence extraction to generate a structured JSON case record.

### Example Case Record

| Field | Result |
|---|---|
| Image | ISIC_0029306.jpg |
| Predicted class | nv |
| nv probability | 98.3% |
| Melanoma (mel) probability | 0.8% |
| Extracted entities | 1 DATE |
| Urgent terms | None detected |
| Final system decision | ROUTINE |

The pipeline combines image probabilities, extracted text entities, structured fields, and decision reasons into a JSON record.

The example demonstrates how image and text evidence can be brought together in one transparent case record.

## 6. Privacy Protection

Implemented Personally Identifiable Information (PII) masking before text extraction and case-record generation.

| Personal identifier | Masking result |
|---|---|
| CNIC | Successful |
| Mobile number | Successful |
| Email address | Successful |

All three sample privacy checks returned `NOT FOUND` in the generated JSON output.

Masking before extraction helps prevent sensitive personal information from appearing in extracted evidence or shared records.

## 7. Limitations and Future Improvements

- Improve medical-domain NER using more representative annotated data.
- Add dermatology-specific warning terms to the text evidence component.
- Improve recognition of disease names, medical datasets, and laboratory terminology.
- Evaluate OCR on real photographed medical documents.
- Improve the multimodal decision logic to flag disagreements between image predictions and clinical notes.
- Conduct further testing before considering any clinical application.

**Disclaimer:** This project is an educational decision-support demonstration. Model predictions are not medical diagnoses and must not replace assessment by qualified healthcare professionals.
**Note:** All numerical results in this README are based on the outputs recorded during my Week 7 experiments. The OCR result (CER = 0.000) was obtained using a clean, synthetic invoice and may differ when processing real-world documents. The multimodal pipeline's `ROUTINE` decision is a rule-based output, not a clinical assessment. The text extractor also incorrectly classified the phrase "the past two months" as a DATE entity, although it represents a duration. The model is intended for educational decision-support purposes only and must not replace evaluation by a qualified healthcare professional.
