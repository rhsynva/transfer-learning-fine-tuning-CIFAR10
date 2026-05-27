# Transfer Learning and Fine-Tuning on CIFAR-10 Using MobileNetV2

This repository contains my implementation of a Deep Learning workflow utilizing Transfer Learning and Fine-Tuning principles to classify images from the classic CIFAR-10 dataset. Rather than training a deep convolutional neural network from scratch—which demands extensive computational time and resources—I leveraged a powerful, pre-trained architecture and adapted it to a multi-class image classification task. Stacking custom dense layers on top of a frozen feature extractor allows the model to leverage rich, pre-learned visual representations (such as edges, textures, and shapes) and successfully adapt them to new target classes.

### Technologies Used

1. Python 3
2. TensorFlow and Keras for designing, compiling, and training the neural network
3. NumPy for multi-dimensional matrix operations and data processing
4. Matplotlib for plotting and analyzing historical training metrics

### Project Workflow

1. Data Loading and Normalization: I loaded the CIFAR-10 dataset directly through Keras, yielding 50,000 training images and 10,000 test images distributed across 10 distinct classes. Each image has an original dimensional shape of (32, 32, 3). To ensure optimal network convergence, I cast the image pixels to float32 and normalized them from the standard [0, 255] range down to a normalized scale of [0.0, 1.0].
2. Label Preprocessing: Because this is a 10-class categorical classification problem, the integer class labels need to be properly formatted for the output layer's activation mechanics. I applied one-hot encoding using the `to_categorical` utility function, converting the label vectors into a binary matrix representation.
3. Base Model Configuration: I selected MobileNetV2 pre-trained on the ImageNet dataset as the underlying backbone network. By setting `include_top=False` during initialization, I discarded the final fully-connected classification layer of MobileNetV2 to make room for a custom classifier designed for the 32x32 pixel dimension inputs.
4. Freezing Weights (Feature Extraction): To preserve the intricate feature representations learned from the massive ImageNet dataset, I set `base_model.trainable = False`. This freezes the operational weights of all the convolutional layers within MobileNetV2, preventing them from updating and getting corrupted during the initial feature extraction phase.
5. Custom Classifier Architecture: I constructed a new network sequence by connecting the frozen base model to a custom neural network head. First, a `GlobalAveragePooling2D` layer reduces the spatial dimensions and downsizes the overall parameter count. Next, a `Dense` layer with 128 units and a ReLU activation function introduces non-linear learning capability, followed by a `Dropout` layer set at a rate of 0.3 to reduce overfitting. Finally, a `Dense` layer with 10 units and a Softmax activation function outputs individual probability distributions across the 10 target classes.
6. Model Compilation and Optimization: The model was compiled using the `Adam` optimizer with a learning rate of 0.001. I designated `categorical_crossentropy` as the loss function given the one-hot encoded state of our target vectors, tracking standard `accuracy` as the core evaluation metric.
7. Training and Evaluation: I trained the transfer learning network for 10 epochs using a batch size of 64, feeding the normalized validation data concurrently to evaluate real-time testing performance. Over the course of 10 epochs, the loss systematically minimized as the validation metrics steadied, establishing a baseline classification framework on the target dataset.
8. Performance Visualization: I used Matplotlib to generate a dual-subplot evaluation layout. The charts map out the training accuracy versus validation accuracy along with training loss versus validation loss over each operational epoch, providing clear, visual confirmation of the training trends and model stability.
