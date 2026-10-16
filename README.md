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
- **Next — Week 6:** Apply transfer learning to a new image-classification domain using a suitable dataset.
