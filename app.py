# Import FastAPI class to create an API application
from fastapi import FastAPI

# Import BaseModel from Pydantic for data validation and schema creation
from pydantic import BaseModel

# Import joblib to load the trained ML model saved earlier
import joblib

# Import numpy for numerical operations
import numpy as np

from fastapi.responses import HTMLResponse
from fastapi import Form



# -------------------------------------------
# Create the FastAPI application instance
# This 'app' object will handle routes and API behavior
# -------------------------------------------
app = FastAPI()


# -------------------------------------------
# Load the saved ML model
# This file must be in the same folder as this script
# -------------------------------------------
model =  joblib.load("iris_model.pkl")



# -------------------------------------------
# Define what input format the API expects
# Pydantic automatically:
# - Validates incoming JSON
# - Converts types (str → float)
# - Throws errors if fields are missing
# -------------------------------------------
class IrisInput(BaseModel):
    sepal_length: float      # Required field: sepal length
    sepal_width: float       # Required field: sepal width
    petal_length: float      # Required field: petal length
    petal_width: float       # Required field: petal width



# -------------------------------------------
# Create a POST endpoint named "/predict"
# This endpoint will:
# 1. Accept input as JSON -> Converted to IrisInput object
# 2. Convert data to NumPy array
# 3. Make prediction using the ML model
# 4. Return predicted class name
# -------------------------------------------
@app.post("/predict")
def predict(data: IrisInput):
    
    # Convert the validated Pydantic object into a 2D NumPy array
    features = np.array([
        [
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]
    ])

    # Predict the class using the saved model
    pred = model.predict(features)[0]

    # Mapping numeric prediction to actual flower names
    classes = ["setosa", "versicolor", "virginica"]

    # Return prediction as JSON response
    return {"prediction": classes[pred]}



# -------------------------------------------
# Health check endpoint:
# Used by developers, monitoring tools, cloud services
# Helps verify that API server is alive and responding
# -------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}



# -------------------------------------------
# Model version endpoint:
# Useful when you update your model later
# This helps determine which version is currently deployed
# -------------------------------------------
@app.get("/version")
def version():
    return {"model_version": "1.0.0"}

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Iris Classifier</title>
        </head>
        <body style="font-family: Arial; padding: 40px;">
            <h2>Iris Flower Prediction</h2>

            <form action="/predict-ui" method="post">
                Sepal Length: <input type="number" step="0.1" name="sepal_length"><br><br>
                Sepal Width: <input type="number" step="0.1" name="sepal_width"><br><br>
                Petal Length: <input type="number" step="0.1" name="petal_length"><br><br>
                Petal Width: <input type="number" step="0.1" name="petal_width"><br><br>

                <button type="submit">Predict</button>
            </form>
        </body>
    </html>
    """
@app.post("/predict-ui", response_class=HTMLResponse)
def predict_ui(
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)
):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred = model.predict(features)[0]

    classes = ["setosa", "versicolor", "virginica"]

    return f"""
    <html>
        <body style="font-family: Arial; padding: 40px;">
            <h2>Prediction Result</h2>
            <p><b>Predicted class:</b> {classes[pred]}</p>
            <a href="/">Try again</a>
        </body>
    </html>
    """
