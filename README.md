# Cats and Dogs Classifier

This project demonstrates how to build an image classifier to distinguish between cats and dogs using deep learning techniques. It explores both building a model from scratch and using transfer learning with a pre-trained VGG16 model to improve performance.

## Project Overview

The project is part of the DIO/BairesDev Machine Learning Practitioner Bootcamp. It aims to highlight the following key areas:

- **Dataset Management**: Handling large image datasets, preprocessing images, and splitting data into training, validation, and test sets.
- **Model Development**: Creating and training a convolutional neural network (CNN) for binary image classification.
- **Transfer Learning**: Fine-tuning a pre-trained VGG16 model to leverage prior knowledge and achieve better results.
- **Evaluation**: Analyzing model performance using metrics such as loss and accuracy and visualizing the training process.

## Key Features

1. **Data Preprocessing**:
   - Images are resized to fit the input size of the model (224x224).
   - Data is normalized and one-hot encoded for efficient processing.

2. **Initial Model**:
   - A custom CNN is built and trained from scratch.
   - The model uses layers such as `Conv2D`, `MaxPooling2D`, and `Dense`.

3. **Transfer Learning**:
   - The VGG16 model is imported with pre-trained weights.
   - A custom classification layer is added, and all but the last layer of VGG16 are frozen.
   - Fine-tuning is performed to adjust the new layer to the dataset.

4. **Evaluation and Visualization**:
   - Training and validation loss and accuracy are plotted for both models.
   - The final test accuracy and loss are computed and displayed.

## Dependencies

This project uses the following libraries:

- Python 3.x
- NumPy
- Keras
- TensorFlow
- Matplotlib
- scikit-learn

Ensure you have these libraries installed before running the notebook.

## Getting Started

1. **Download the Dataset**:
   - The dataset is the Microsoft Cats vs. Dogs dataset, available on Kaggle.
   - It is downloaded and extracted automatically within the notebook.

2. **Run the Notebook**:
   - Open the notebook and execute each cell in sequence.
   - The code handles data loading, preprocessing, training, and evaluation.

3. **Experiment**:
   - Modify the architecture or hyperparameters to explore their effects on performance.
   - Try other pre-trained models or datasets for further practice.

## Results

The final results include:

- **Accuracy**: The test accuracy achieved with the fine-tuned model.
- **Visualizations**: Loss and accuracy curves for both the custom CNN and the transfer learning approach.
- **Prediction**: The model can classify new images of cats and dogs with high accuracy.

## Notes

- The notebook contains comments to guide you through the code and explain key concepts.
- It is designed for educational purposes and can be extended to multi-class classification or other datasets.

## License

This project is for educational purposes under the DIO/BairesDev Machine Learning Bootcamp. Please refer to the dataset's Kaggle license for usage terms.

---

Feel free to reach out if you have questions or want to discuss the implementation further!

