
# Heart Disease Prediction Project

## Overview

This project aims to predict the likelihood of heart disease in patients using machine learning.  It explores, analyzes, and builds classification models using the  [Heart Disease UCI](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) dataset.  The project covers data exploration, preprocessing, model selection, hyperparameter tuning, and model evaluation.

## Data Source

The dataset used in this project is the Heart Disease UCI dataset, sourced from the UCI Machine Learning Repository.  It contains various patient attributes and a binary target variable indicating the presence or absence of heart disease.  The specific file used is `heart-disease pj.csv`.

## Features

The dataset includes the following features:

* `age`: Patient's age
* `sex`: Patient's sex (1 = male, 0 = female)
* `cp`: Chest pain type
* `trestbps`: Resting blood pressure
* `chol`: Serum cholesterol
* `fbs`: Fasting blood sugar (> 120 mg/dl)
* `restecg`: Resting electrocardiographic results
* `thalach`: Maximum heart rate achieved
* `exang`: Exercise-induced angina
* `oldpeak`: ST depression induced by exercise relative to rest
* `slope`: The slope of the peak exercise ST segment
* `ca`: Number of major vessels colored by fluoroscopy
* `thal`: Thallium scan result
* `target`:  Presence of heart disease (1 = yes, 0 = no)

## Getting Started

### Prerequisites

* Python 3.x
* Libraries:
    * pandas
    * numpy
    * matplotlib
    * seaborn
    * scikit-learn
    * Jupyter Notebook (Optional, for running the notebook)

### Installation

1.  Clone the repository:
    ```bash
    git clone <your_repository_url>
    cd heart-disease-prediction
    ```
2.  It's recommended to create a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    (You may need to create a `requirements.txt` file.  A basic one is shown below.  It's best to generate this using `pip freeze > requirements.txt` in your virtual environment.)

    **Example `requirements.txt`:**
    ```
    pandas
    numpy
    matplotlib
    seaborn
    scikit-learn
    ```

## Usage

1.  **Data Preparation:** The data is read from `heart-disease pj.csv`.  Ensure this file is in the `Dataset` directory.
2.  **Exploratory Data Analysis (EDA):** The Jupyter Notebook (`<your_notebook_name>.ipynb`)  contains EDA, including data exploration, visualization, and correlation analysis.
3.  **Model Training:** The notebook trains and evaluates the following models:
    * Logistic Regression
    * K-Nearest Neighbors (KNN)
    * Random Forest Classifier
4.  **Model Evaluation:** The notebook evaluates the models using metrics such as:
    * Accuracy
    * ROC Curve and AUC
    * Confusion Matrix
    * Classification Report
    * Precision, Recall, and F1-score (with cross-validation)
5. **Running the code**
    * Run the jupyter notebook and execute all the cells.
    * The notebook performs  Exploratory Data Analysis (EDA), model training, and evaluation.
    * The best performing model is selected.

## Model Evaluation

The Logistic Regression model achieved the best performance in this project. Key evaluation metrics include:

* Accuracy:  88.52% on the test set.
* Cross-validated Accuracy: 84.47%
* Precision: 82.08%
* Recall: 92.12%
* F1-Score: 86.73%

The ROC curve and confusion matrix are available in the notebook.  Feature importance analysis highlights the key factors influencing heart disease prediction.

## Results

The Logistic Regression model provides a good balance of precision and recall.

## Further Improvements

* Explore more advanced feature engineering techniques.
* Experiment with other machine learning models or ensemble methods.
* Perform more extensive hyperparameter tuning.
* Deploy the model as a web application.
* Investigate the impact of different data scaling methods.

## Author

Raja Kannan
