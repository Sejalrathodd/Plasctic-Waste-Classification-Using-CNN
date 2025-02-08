# Plastic-Waste-Classification-Using-CNN

## Project Description
Plastic pollution is a growing concern globally, and effective waste segregation is critical to tackling this issue. This project employs a CNN model to classify plastic waste into distinct categories, facilitating automated waste management.

---

## Dataset
The dataset used for this project is the **Waste Classification Data** by Sashaank Sekar. It contains a total of **25,077 labeled images**, divided into two categories: Organic and Recyclable. This dataset is designed to facilitate waste classification tasks using machine learning techniques.

### Key Details:
- **Total Images:** 25,077  
- **Training Data:** 22,564 images (85%)  
- **Test Data:** 2,513 images (15%)  
- **Classes:** Organic and Recyclable  
- **Purpose:** To aid in automating waste management and reducing the environmental impact of improper waste disposal.

---

## Approach:
1. Studied waste management strategies and white papers.
2. Analyzed the composition of household waste.
3. Segregated waste into two categories (Organic and Recyclable).
4. Leveraged IoT and machine learning to automate waste classification.

---

## Dataset Link:
You can access the dataset here: [Waste Classification Data](https://www.kaggle.com/datasets/techsash/waste-classification-data).

**Note:** Ensure appropriate dataset licensing and usage guidelines are followed.

---

## Weekly Progress
This section will be updated weekly with progress details and corresponding Jupyter Notebooks.

### Week 1: Libraries, Data Import, and Setup  
**Date:** 20th January 2025 - 27th January 2025  

**Activities:**  
- Imported the required libraries and frameworks.  
- Set up the project environment.  
- Explored the dataset structure.

**Notebooks:**  
- [Week1-Libraries-Importing-Data-Setup.ipynb](importing-data-week1.ipynb)  
- [Kaggle Notebook](https://www.kaggle.com/code/rathodsejal/importing-data-week1)

---

### Week 2:Week 2: Model Training, Evaluation, and Predictions
Date: 28th January 2025 - 31st January 2025

**Activities:**

- Trained a Convolutional Neural Network (CNN) model on the dataset.
- Optimized hyperparameters to enhance accuracy.
- Evaluated model performance using accuracy and loss metrics.
- Conducted predictions on test images.
- Visualized classification results with a confusion matrix.

**Notebooks:**

- Week2-Model-Training-Evaluation-Predictions.ipynb
- Kaggle Notebook

**Conclusion & Summary of Model Performance**

**1. Overview of the Model**

The trained CNN model was designed to classify waste into two categories:

- O (Organic Waste)
- R (Recyclable Waste)

The model architecture included convolutional layers, max-pooling, batch normalization, and fully connected dense layers. It was optimized using categorical cross-entropy loss and evaluated based on accuracy.

**2. Model Evaluation on Test Data**

After training, the model was evaluated on a separate test dataset with the following results:

- **Test Accuracy:** 85.32%
- **Test Loss:** 0.3997

This indicates that the model correctly classifies waste in approximately 85 out of 100 cases on unseen data. The relatively low test loss suggests that the model has learned meaningful patterns and is not significantly overfitting.

**3. Predictions and Sample Results**

The model made predictions on test images, converting probability outputs into class labels. A sample comparison of predicted vs. actual results is as follows:

- **Predicted Classes:** ['O', 'O', 'O', 'O', 'R', 'O', 'O', 'O', 'O', 'O']
- **Actual Classes:** ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

The model mostly predicted correctly, but one case (index 5) was classified as "R" instead of "O", indicating a potential misclassification.

**4. Classification Report Analysis**

The classification report provides deeper insights into model performance for each class:

| Class             | Precision | Recall | F1-Score | Support |
|-------------------|-----------|--------|----------|---------|
| O (Organic)       | 0.89      | 0.84   | 0.86     | 1401    |
| R (Recyclable)    | 0.81      | 0.87   | 0.84     | 1112    |
| **Overall Accuracy** | 0.85   |        |          |         |
| **Macro Avg**     | 0.85      | 0.85   | 0.85     | 2513    |
| **Weighted Avg**  | 0.85      | 0.85   | 0.85     | 2513    |

**Key Observations:**

- **Precision for Organic (O) is higher:** The model is more confident when predicting Organic waste than Recyclable waste.
- **Recall for Recyclable (R) is higher:** The model captures more actual Recyclable waste but sometimes mislabels Organic waste as Recyclable.
- **Balanced F1-score:** The model performs well for both classes, with a slight bias towards correctly predicting Organic waste.

**5. Confusion Matrix Insights**

The confusion matrix helps visualize the model’s errors:

- **1401 Organic waste samples:**
  - 1180 were correctly classified as Organic (True Positives)
  - 221 were misclassified as Recyclable (False Negatives)
- **1112 Recyclable waste samples:**
  - 968 were correctly classified as Recyclable (True Positives)
  - 144 were misclassified as Organic (False Positives)

**Key Takeaways from Confusion Matrix:**

- The model performs well overall but struggles slightly more with distinguishing Recyclable waste from Organic waste.
- 221 Organic samples were wrongly classified as Recyclable waste, which might be due to overlapping features (e.g., food-contaminated paper/cardboard).

**Final Conclusion:**

The CNN model has achieved 85.32% accuracy, which is a strong performance for waste classification. While the model performs slightly better for Organic waste, it can sometimes misclassify Recyclable waste.

**Possible Improvements:**

- Adding more diverse training data to reduce misclassification.
- Applying data augmentation to expose the model to more variations.
- Experimenting with fine-tuning on pre-trained CNN models (like ResNet, VGG16) to further boost accuracy.
- Tweaking hyperparameters (learning rate, dropout rate) to optimize performance.


### Week 3: Week 3

Streamlit App and Model Deployment
Date: 4th February 2025 - 7th February 2025

Activities:

Developed a Streamlit web application for real-time waste classification.
Uploaded the trained model to Kaggle and GitHub for public access.
Finalized the project documentation and README formatting.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.
