# Transfer Learning and Fine-Tuning on CIFAR-10 Using MobileNetV2

This repository contains my implementation of a transfer learning and fine-tuning workflow that classifies images from the CIFAR-10 dataset using an ImageNet-pretrained MobileNetV2. Instead of training a deep convolutional network from scratch, I reuse a pretrained backbone and adapt it to a 10-class image classification task in two stages: feature extraction with a frozen backbone, followed by fine-tuning of the top layers.

## Results

| Stage | Accuracy |
|---|---|
| Stage 1: frozen backbone (10 epochs) | 87.1% validation (best: 88.0%) |
| Stage 2: fine-tuning (10 epochs) | 90.0% validation |
| **Final evaluation on the test set** | **89.41%** |

A first version of this project, which fed 32×32 images directly into the network, reached only ~36% validation accuracy. See "Problems Found and Fixed" below.

## Technologies Used

1. Python 3
2. TensorFlow and Keras for building, compiling, and training the model
3. NumPy for array operations
4. Matplotlib for plotting training curves and the confusion matrix

## Project Workflow

1. **Data Loading:** CIFAR-10 is loaded through Keras: 50,000 training and 10,000 test images across 10 classes, each of shape (32, 32, 3). Pixels are cast to `float32` and kept in the [0, 255] range, because rescaling is done inside the model.
2. **Label Preprocessing:** Integer labels are one-hot encoded with `to_categorical`.
3. **Train/Validation Split:** The training set is split into 45,000 images for training and 5,000 for validation. The test set is not used during training or model selection.
4. **Base Model:** MobileNetV2 pretrained on ImageNet, loaded with `include_top=False` and `input_shape=(96, 96, 3)`, a size for which pretrained ImageNet weights are available.
5. **Model Architecture:** The model takes 32×32 inputs and applies a `Resizing` layer (32×32 → 96×96) and a `Rescaling` layer that maps pixels from [0, 255] to [-1, 1], which is the range MobileNetV2 expects. Then come the frozen MobileNetV2 backbone, `GlobalAveragePooling2D`, a `Dense` layer (128 units, ReLU), `Dropout` (0.3), and a `Dense` softmax layer with 10 units.
6. **Stage 1, Feature Extraction:** The backbone is frozen (`base_model.trainable = False`). The model is compiled with `Adam` (learning rate 0.001) and `categorical_crossentropy`, and trained for 10 epochs with batch size 64.
7. **Stage 2, Fine-Tuning:** The backbone is unfrozen, then the first 100 of its 154 layers are frozen again (`fine_tune_at = 100`), so only the last 54 layers are trained. The model is recompiled with a much lower learning rate (0.00001) and trained for another 10 epochs.
8. **Evaluation:** The model is evaluated on the test set once, at the end, giving 89.41% accuracy. A confusion matrix shows where the model makes mistakes; the most visible confusion is between **cat** and **dog**.
9. **Visualization:** Matplotlib is used for training/validation accuracy and loss curves for each stage, and for the confusion matrix.

## Problems Found and Fixed

The first version of this notebook reached only ~36% accuracy. Debugging showed four issues:

1. **Input size:** 32×32 images were fed to a network whose ImageNet weights are meant for larger inputs. Fixed by resizing to 96×96 inside the model.
2. **Input scaling:** Pixels were scaled to [0, 1], while MobileNetV2 expects [-1, 1]. Fixed with a `Rescaling` layer.
3. **Evaluation leakage:** The test set was used as validation data. Fixed with a separate 45k/5k train/validation split, using the test set only for the final evaluation.
4. **Fine-tuning did not actually happen:** `fine_tune_at` was equal to the total number of layers (154), so every layer stayed frozen. Fixed by setting it to 100 and lowering the learning rate to 1e-5.

A plotting bug (the fine-tuning loss curve used the stage 1 history) was fixed as well.

## Observations

- Stage 1 shows mild overfitting: validation loss stops improving after about epoch 5 while training accuracy keeps rising.
- Fine-tuning improved validation accuracy from about 87% to 90%.
- The weakest classes are cat and dog, which are often confused with each other.
