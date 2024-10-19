# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder  # For converting string labels to numerical values
from sklearn.svm import SVC  # Support Vector Classification model
from keras.src.models import Sequential  # For creating a sequential model in Keras
from keras.src.layers import Dense  # For adding layers to the model
from keras.src.utils import to_categorical  # For converting labels to categorical format

# Initialize the LabelEncoder to convert class labels to numerical format
label_encoder = LabelEncoder()

# Load the Iris dataset from the provided URL
# The dataset does not have headers, so we specify header=None
iris = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/refs/heads/master/iris.csv", header=None)

# Rename the columns for better readability
iris.columns = ["sepal length", "sepal width", "petal length", "petal width", "Class"]

# Convert the class labels (strings) to numerical values using LabelEncoder
# This is necessary for models that work with numerical data
iris["Class"] = label_encoder.fit_transform(iris["Class"])

# Display the first few rows of the dataset to verify the changes
print(iris.head())

# Separate the features (X) and the target variable (y)
y = iris["Class"].values  # Target variable: class labels
X = iris.drop("Class", axis=1).values  # Features: all columns except 'Class'

# Initialize the Support Vector Classifier model
svm_model = SVC()

# Fit the model to the training data
svm_model.fit(X, y)

# Evaluate the model's performance on the training data
accuracy = svm_model.score(X, y)
print(f"SVM Model Accuracy: {accuracy}")

# Prepare the output labels for the neural network
# Convert the integer labels to one-hot encoded format
y_categorical = to_categorical(y)

# Initialize a sequential neural network model
neural_model = Sequential()

# Add the first hidden layer with 5 neurons and 4 input features
# The Dense layer is fully connected
neural_model.add(Dense(units=5, input_dim=4))  # 4 inputs connected to 5 neurons

# Add the output layer with 3 neurons for the 3 classes, using softmax activation
# Softmax converts outputs to probabilities
neural_model.add(Dense(units=3, activation="softmax"))

# Compile the model with the Adam optimizer and categorical crossentropy loss
# This prepares the model for training
neural_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the neural network model on the data for 200 epochs
# verbose=1 means that training progress will be displayed
neural_model.fit(X, y_categorical, epochs=200, verbose=1)
