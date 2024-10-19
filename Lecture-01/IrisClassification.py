import pandas as pd
from sklearn.preprocessing import LabelEncoder # This function will sort by alphabet

Luka = LabelEncoder()

# We have x input and y output
# x are all the data 1-2-3 and y is 4
iris = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/refs/heads/master/iris.csv", header = None)

# Naming columns
iris.columns = ["sepal length", "sepal width", "petal length", "petal width", "Class"]

# As we can't work with text, we should convert strings to numbers: in this case classes
# Input should be a number and output number as well
iris["Class"] = Luka.fit_transform(iris["Class"])

iris.head()

y = iris["Class"].values
X = iris.drop("Class", axis = 1).values # drop is deleting

# SVM
# C as Classification (SVC)
# R as Regression (SVR)
from sklearn.svm import SVC
Model = SVC()
Model.fit(X,y)
Model.score(X,y)

# import keras
from keras.src.models import Sequential # Going in a sequential way, MLP
from keras.src.layers import Dense # Every layer connected to every layer in MLP
from keras.src.utils import to_categorical # Transform everything into binary format

# not it's 0-2, and it is required to be binary, using to_categorical
# y

y = to_categorical(y)
# y

model = Sequential()

# Add one inside layer
# 4 inputs connected to 5 layers
model.add(Dense(units = 5, input_dim = 4)) # First 2 vertical lines

# Writing output - 3 outputs
model.add(Dense(units = 3, activation = "softmax")) # softmax - every answer will be given as probability

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(X,y,epochs=200,verbose=1) # verbose = tell me everything
