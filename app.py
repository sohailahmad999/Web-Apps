# Importing the Libraries
import streamlit as st
# data processing
import pandas as pd
# data visualization
import seaborn as sns
import matplotlib.pyplot as plt
# Algorithms
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 

# make contriners
header = st.container()
datasets = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title("Titanic App (Kashti App)")
    st.text("RMS Titanic was a British passenger liner, operated by the White Star Line, which sank in the North Atlantic Ocean on 15 April 1912 after striking an iceberg during her maiden voyage from Southampton, UK, to New York City, United States.")
    st.text("This is an app to predict the survival of the Titanic passengers.")
 
with datasets:
    st.header("Titanic Dataset")
    st.text("We will work on Titanic dataset")
    # import data
    df = sns.load_dataset('titanic')
    df = df.dropna() # drop missing values
    st.write(df.head())


    st.subheader("Number of Passengers By Gender")
    st.bar_chart(df['sex'].value_counts())

    st.subheader("Number of Passengers By Man, Woman and Child")
    st.bar_chart(df['who'].value_counts())


    #other plots
    st.subheader("Number of Passengers by Class")
    st.bar_chart(df['class'].value_counts())
    # barplot
    st.subheader("Number of Passengers by Age")
    st.bar_chart(df['age'].sample(10)) # or head(10)
    
    st.subheader("Number of Passengers by Fare")
    st.bar_chart(df['fare'].sample(10)) # or head(10)
with features:
    st.header("App Features")
    st.text("Features of the App:")
    st.markdown('1. **Surival Prediction :** Predict the survival of the passengers')
    st.markdown('2. **Accuracy Calculation:** Calculate the accuracy of the model by calculating Mean Squared Error, Mean Absolute Error and R2 Score')


with model_training:
    st.header("Parameters Selection")
    # making columns
    input, display = st.columns(2)

    # Passenger Selection
    max_depth= input.slider("Passengers' selection: ", min_value=1, max_value=182, value=10,step=1)
    
# n_estimators
n_estimators = input.selectbox("Number of trees required?", options=[50,100,150,200,250,300,'No limit'])

# input features from user
input_features = input.selectbox("Feature's Selection: ", options=['pclass','sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'class', 'who']) 

# machine learning model
model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

# define x and y
x = df[[input_features]]
y = df[['survived']]

# fit and predict model
model.fit(x,y)
pred = model.predict(x)

# visualize results
display.subheader("Passengers' Survival Prediction")
display.write(pred)


# display the accuracy
display.subheader("Accuracy")
display.subheader("Mean absolute error of the model is: ")
display.write(mean_absolute_error(y,pred))
display.subheader("Mean squared error of the model is: ")
display.write(mean_squared_error(y,pred))
display.subheader("R squared error of the model is: ")
display.write(r2_score(y,pred))

