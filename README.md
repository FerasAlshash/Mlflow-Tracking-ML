# Drug Classification with MLflow

This project demonstrates how to use **MLflow** for experiment tracking and model management in a drug classification task. The goal is to classify the type of drug that should be prescribed based on patient data.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

In this project, we explore multiple machine learning models (Random Forest, Logistic Regression, Decision Tree, K-Nearest Neighbors, and Naive Bayes) to classify drugs based on patient attributes such as age, sex, blood pressure, cholesterol levels, and sodium-to-potassium ratio. 

We use **MLflow** to:
- Track experiments and log hyperparameters.
- Log evaluation metrics like accuracy, precision, recall, and F1-score.
- Save trained models for future deployment.

The dataset used in this project is `drug200.csv`, which contains 200 samples of patient data.

---

## Features

- **Experiment Tracking**: Use MLflow to track experiments and compare results across different models.
- **Pipeline Integration**: Preprocessing (scaling and encoding) integrated into a scikit-learn pipeline.
- **Model Comparison**: Evaluate multiple models and their performance metrics.
- **Local MLflow Server**: Run an MLflow server locally to visualize results via the MLflow UI.

---

## Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.7 or higher
- Pip (Python package installer)
- MLflow library
- Jupyter Notebook (optional, for exploration)

---

