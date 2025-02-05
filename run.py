
import mlflow.sklearn
import pandas as pd
 
 
logged_model = "runs:/34ee71a327b14444a87bd1d95a300fbe/Decision Tree"
 
# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)
 
# Age,Sex,BP,Cholesterol,Na_to_K
 
data = pd.DataFrame({
    'Age': [23],
    'Sex': ['F'],
    'BP': ['HIGH'],
    'Cholesterol': ['HIGH'],
    'Na_to_K': [25.355]
})
 
prediction = loaded_model.predict(data)
print(prediction)
 