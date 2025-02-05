import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline


mlflow.set_tracking_uri("http://127.0.0.1:8000")

def main(model_name, model, params, test_size, random_state):
    # Load the dataset
    df = pd.read_csv('drug200.csv')
    X = df.drop('Drug', axis=1)
    y = df['Drug']

   

    # Define features to scale or encode
    num = ['Age', 'Na_to_K']
    cat = ['Sex', 'BP', 'Cholesterol']

    # Preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num),
        ('cat', OneHotEncoder(), cat)
    ])

    # Define the full pipeline including model
    pipeline = Pipeline([
        ('preprocessor', preprocessor), 
        ('model', model)
    ])

    # Split dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    with mlflow.start_run():
        print(f'Running model: {model_name}')
        mlflow.log_param('model_name', model_name)
        
        # Log model hyperparameters
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        
        # Train the model
        pipeline.fit(X_train, y_train)

        # Predict on test set
        y_pred = pipeline.predict(X_test)
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)

        # Log evaluation metrics
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('f1_score', f1)

        # Log model to MLflow
        mlflow.sklearn.log_model(pipeline, 'model')

        # Print results
        print(f'Model: {model_name}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')


# Run the experiment
if __name__ == '__main__':

    # Set the experiment in MLflow
    mlflow.set_experiment('drug_classification')
    
    models = [
        ('Random Forest', RandomForestClassifier(), {"n_estimators": 30, "criterion": "entropy"}),
        ('Logistic Regression', LogisticRegression(), {"C": 0.1}), 
        ('Decision Tree', DecisionTreeClassifier(), {"criterion": "entropy"}), 
        ('K Nearest Neighbors', KNeighborsClassifier(), {"n_neighbors": 3}), 
        ('Naive Bayes', GaussianNB(), {})
    ]
    
    test_size = 0.2
    random_state = 42

    # Iterate over all models
    for model_name, model, params in models:
        main(model_name, model, params, test_size, random_state)
