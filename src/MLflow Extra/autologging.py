import mlflow

# Set the tracking URI to the server URI
mlflow.set_tracking_uri(uri = "http://127.0.0.1:8080")

# Auto Logging
"""
•  Metrics - MLflow pre-selects a set of metrics to log, based on what model and library you use
•  Parameters - hyper params specified for the training, plus default values provided by the library if not explicitly set
•  Model Signature - logs Model signature instance, which describes input and output schema of the model
•  Artifacts - e.g. model checkpoints
•  Dataset - dataset object used for training (if applicable), such as tensorflow.data.Dataset
"""
mlflow.autolog()

# Import required libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the model hyperparameters
params = {
    "max_depth": 15,
    "random_state": 42
}

# Setting experiment
mlflow.set_experiment("decision tree")

# Running the experiment
with mlflow.start_run():
    # Train the model
    dt = DecisionTreeClassifier(**params)
    dt.fit(X_train, y_train)

    # Predict on the test set
    y_pred = dt.predict(X_test)

    # Logging Plots (Confusion Matrix, Precision-Recall Curve, ROC Curve) - Not logged by autologging
    
    # Setting tags
    mlflow.set_tags(
        {
            "name": "Dexter Morgan",
            "Subject": "MLOps",
            "Tool": "MLFlow"
        }
    )