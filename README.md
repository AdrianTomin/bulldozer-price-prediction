# Predicting the Sale Price of Bulldozers Using Machine Learning

In this project, we aim to predict the sale price of bulldozers using machine learning techniques. The data used for this project is from the Kaggle Bluebook for Bulldozers competition.

Built with:
<br>
<br>
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) 
<br>
<br>
![Jupyter Notebook](https://img.shields.io/badge/jupyter-fff.svg?style=for-the-badge&logo=jupyter&logoColor=orange)
<br>
<br>
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
<br>
<br>
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) 
<br>
<br>
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) 
<br>
<br>
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
<br>
<br>

## 1. Problem Definition
> How well can we predict the future sale price of a bulldozer given its characteristics and previous examples of how much similar bulldozers have been sold for?

## 2. Data
The data is downloaded from the Kaggle Bluebook for Bulldozers competition. There are three main datasets:

* `Train.csv` is the training set, which contains data through the end of 2011.
* `Valid.csv` is the validation set, which contains data from January 1, 2012 - April 30, 2012. Your score on this set is used to create the public leaderboard.
* `Test.csv` is the test set, which contains data from May 1, 2012 - November 2012. Your score on the test set determines your final rank for the competition.

## 3. Evaluation
The evaluation metric for this project is the RMSLE (root mean squared log error) between the actual and predicted auction prices.

For more details on the evaluation of this project, check: https://www.kaggle.com/c/bluebook-for-bulldozers/overview/evaluation

## 4. Features
Kaggle provides a data dictionary detailing all the features of the dataset. You can view this data dictionary in `data/bluebook-for-bulldozers/Data Dictionary.xlsx`.

## 5. Project Steps
The main steps we go through in this project are:

1. **Exploratory Data Analysis (EDA)**: Understanding the data by visualizing and identifying patterns.
2. **Data Preprocessing**: Cleaning the data, handling missing values, and converting categorical variables into numerical ones.
3. **Feature Engineering**: Creating new features from existing ones to help the model learn better.
4. **Model Training**: Training machine learning models to predict bulldozer prices.
5. **Model Evaluation**: Evaluating the performance of the trained models using RMSLE.
6. **Hyperparameter Tuning**: Improving model performance by tuning hyperparameters.
7. **Making Predictions**: Using the trained model to make predictions on the test data.

## 6. Installation

### 6.1 Prerequisites

- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) (Anaconda or Miniconda distribution)
- [Git](https://git-scm.com/)

### 6.2 Installing Dependencies
#### Option 1: Using Conda

1. **Clone the repository:**

    ```bash
    git clone https://github.com/AdrianTomin/bulldozer-price-prediction.git
    cd bulldozer-price-prediction
    ```

2. **Create and activate the Conda environment:**

    ```bash
    conda env create -f environment.yml
    conda activate bulldozer-price-prediction
    ```

#### Option 2: Using pip

1. **Clone the repository:**

    ```bash
    git clone https://github.com/AdrianTomin/bulldozer-price-prediction.git
    cd bulldozer-price-prediction
    ```

2. **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

Choose one of these options to set up the environment, depending on your preference.
`

### 6.3 Setting Up the Environment

1. **Install Jupyter Notebook or JupyterLab:**

    ```bash
    conda install -c conda-forge notebook
    # or for JupyterLab
    conda install -c conda-forge jupyterlab
    ```

2. **Start Jupyter Notebook or JupyterLab:**

    ```bash
    jupyter notebook
    # or for JupyterLab
    jupyter lab
    ```

## 7. Running the Project Locally

1. **Navigate to the project directory:**

    ```bash
    cd bulldozer-price-prediction
    ```

2. **Start the Jupyter Notebook server:**

    ```bash
    jupyter notebook
    ```

3. **Open the notebook:**

    In the Jupyter Notebook interface, open the `bulldozer-price-prediction.ipynb` notebook.

4. **Run the notebook cells:**

    Execute the cells in the notebook to train the model and make predictions. Ensure you have downloaded the dataset and placed it in the appropriate directory as mentioned in the notebook.

---

This README provides a comprehensive guide for setting up the environment, installing dependencies, and running the project locally. Adjust paths and repository links as needed to match your specific setup.


## Badges
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

## Authors
- [@AdrianTomin](https://www.github.com/AdrianTomin)