import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


df = pd.read_csv('diabetes.csv')

#headings
st.title('Diabetes Prediction App')
st.sidebar.header('Patient Data')
st.subheader('Description Stats of Data')
st.write(df.describe())

# Data Split into X and Y and train test split
x = df.drop('Outcome', axis=1)
y = df['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Function to get user input
def user_report():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 2)
    glucose = st.sidebar.slider('Glucose', 0, 199, 110)
    bp = st.sidebar.slider('BloodPressure', 0, 122, 80)
    sk = st.sidebar.slider('SkinThickness', 0, 99, 12)
    insulin = st.sidebar.slider('Insulin', 0.0, 846.0, 80.0)
    bmi = st.sidebar.slider('BMI', 0, 67, 5)
    dpf = st.sidebar.slider('DiabetesPedigreeFunction', 0.07, 2.42, 0.3725)
    age = st.sidebar.slider('Age', 21, 81, 29)
    # Store a dictionary into a variable
    user_report_data = {'pregnancies': pregnancies,
                 'glucose': glucose,
                 'bp': bp,
                 'sk': sk,
                 'insulin': insulin,
                 'bmi': bmi,
                 'dpf': dpf,
                 'age': age}
    # Transform the data into a data frame
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

# patient data
user_data = user_report()  
st.subheader("Patient Data")
st.write('user_data')

# Random Forest Classifier
rc = RandomForestClassifier()
rc.fit(x_train, y_train)
user_result = rc.predict(user_data)

# Visualize the data
st.title('Visualized Patient Data')

#  color function
if user_result[0] == 0: # zero mean index
    color = 'blue'
else:
    color = 'red'
# Age vs Pregnancies
st.header('Pragency Count Graph (Other Vs Yours)') 
fig_preg= plt.figure()
ax1= sns.scatterplot(x='Age', y='Pregnancies',data=df, hue='Outcome', palette='Greens')
ax2= sns.scatterplot(x= user_data['age'], y=user_data['pregnancies'], s=150,color=color)
plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 20, 2))
plt.title('0- Healthy & 1- Diabetic')
st.pyplot(fig_preg) 

# Age vs Glucose
st.header('Age Count Graph (Other Vs Yours)') 
figg_preg= plt.figure()
ax1= sns.scatterplot(x='Age', y='Glucose',data=df, hue='Outcome', palette='Greens')
ax2= sns.scatterplot(x= user_data['age'], y=user_data['glucose'], s=150,color=color)
plt.xticks(np.arange(20, 85, 5))
plt.yticks(np.arange(0, 200, 20))
plt.title('0- Healthy & 1- Diabetic')
st.pyplot(figg_preg) 

# Age vs BloodPressure
st.header('BloodPressure Count Graph (Other Vs Yours)')
fig_bp= plt.figure()
ax1= sns.scatterplot(x='Age', y='BloodPressure',data=df, hue='Outcome', palette='Greens')
ax2= sns.scatterplot(x= user_data['age'], y=user_data['bp'], s=150,color=color)
plt.xticks(np.arange(20, 85, 5))
plt.yticks(np.arange(0, 130, 40))
plt.title('0- Healthy & 1- Diabetic')
st.pyplot(fig_bp)

# Age vs SkinThickness
st.header('SkinThickness Count Graph (Other Vs Yours)')
fig_sk= plt.figure()
ax1= sns.scatterplot(x='Age', y='SkinThickness',data=df, hue='Outcome', palette='Greens')
ax2= sns.scatterplot(x= user_data['age'], y=user_data['sk'], s=150,color=color)
plt.xticks(np.arange(20, 85, 5))
plt.yticks(np.arange(0, 100, 20))
plt.title('0- Healthy & 1- Diabetic')
st.pyplot(fig_sk)

# Age vs Insulin
st.header('Insulin Count Graph (Other Vs Yours)')
fig_ins= plt.figure()
ax1= sns.scatterplot(x='Age', y='Insulin',data=df, hue='Outcome', palette='Greens')
ax2= sns.scatterplot(x= user_data['age'], y=user_data['insulin'], s=150,color=color)
plt.xticks(np.arange(20, 85, 5))
plt.yticks(np.arange(0, 800, 70))
plt.title('0- Healthy & 1- Diabetic')
st.pyplot(fig_ins)

# Age vs BMI
st.header('BMI Count Graph (Other Vs Yours)')
fig_bmi= plt.figure()
ax1= sns.scatterplot(x='Age', y='BMI',data=df, hue='Outcome', palette='Greens')
ax2= sns.scatterplot(x= user_data['age'], y=user_data['bmi'], s=150,color=color)
plt.xticks(np.arange(20, 100, 5))
plt.yticks(np.arange(0, 80, 10))
plt.title('0- Healthy & 1- Diabetic')
st.pyplot(fig_bmi)

# Age vs DiabetesPedigreeFunction
st.header('DiabetesPedigreeFunction Count Graph (Other Vs Yours)')
fig_dpf= plt.figure()
ax1= sns.scatterplot(x='Age', y='DiabetesPedigreeFunction',data=df, hue='Outcome', palette='Greens')
ax2= sns.scatterplot(x= user_data['age'], y=user_data['dpf'], s=150,color=color)
plt.xticks(np.arange(18, 90, 4))
plt.yticks(np.arange(0, 3, 0.5))
plt.title('0- Healthy & 1- Diabetic')
st.pyplot(fig_dpf)

# Pragency vs Glucose
st.header('Pragency vs Glucose Graph (Other Vs Yours)')
fig_pregg= plt.figure()
ax1= sns.scatterplot(x='Pregnancies', y='Glucose',data=df, hue='Outcome', palette='Greens')
ax2= sns.scatterplot(x= user_data['pregnancies'], y=user_data['glucose'], s=150,color=color)
plt.xticks(np.arange(0, 20, 2))
plt.yticks(np.arange(0, 200, 20))
plt.title('0- Healthy & 1- Diabetic')
st.pyplot(fig_pregg)

# Pragency vs BloodPressure
st.header('Pragency vs BloodPressure Graph (Other Vs Yours)')
fig_pregbp= plt.figure()
ax1= sns.scatterplot(x='Pregnancies', y='BloodPressure',data=df, hue='Outcome', palette='Greens')
ax2= sns.scatterplot(x= user_data['pregnancies'], y=user_data['bp'], s=150,color=color)
plt.xticks(np.arange(0, 20, 2))
plt.yticks(np.arange(0, 200, 20))
plt.title('0- Healthy & 1- Diabetic')
st.pyplot(fig_pregbp)

# Pragency vs SkinThickness
st.header('Pragency vs SkinThickness Graph (Other Vs Yours)')
fig_pregsk= plt.figure()
ax1= sns.scatterplot(x='Pregnancies', y='SkinThickness',data=df, hue='Outcome', palette='Greens')
ax2= sns.scatterplot(x= user_data['pregnancies'], y=user_data['sk'], s=150,color=color)
plt.xticks(np.arange(0, 20, 2))
plt.yticks(np.arange(0, 100, 40))
plt.title('0- Healthy & 1- Diabetic')
st.pyplot(fig_pregsk)

# Pragency vs Insulin
st.header('Pragency vs Insulin Graph (Other Vs Yours)')
fig_pregins= plt.figure()
ax1= sns.scatterplot(x='Pregnancies', y='Insulin',data=df, hue='Outcome', palette='Greens')
ax2= sns.scatterplot(x= user_data['pregnancies'], y=user_data['insulin'], s=150,color=color)
plt.xticks(np.arange(0, 20, 2))
plt.yticks(np.arange(0, 900, 50))
plt.title('0- Healthy & 1- Diabetic')
st.pyplot(fig_pregins)

# Pragency vs BMI
st.header('Pragency vs BMI Graph (Other Vs Yours)')
fig_pregbmi= plt.figure()
ax1= sns.scatterplot(x='Pregnancies', y='BMI',data=df, hue='Outcome', palette='Greens')
ax2= sns.scatterplot(x= user_data['pregnancies'], y=user_data['bmi'], s=150,color=color)
plt.xticks(np.arange(0, 20, 2))
plt.yticks(np.arange(0, 60, 5))
plt.title('0- Healthy & 1- Diabetic')
st.pyplot(fig_pregbmi)

# Pragency vs DiabetesPedigreeFunction
st.header('Pragency vs DiabetesPedigreeFunction Graph (Other Vs Yours)')
fig_pregdpf= plt.figure()
ax1= sns.scatterplot(x='Pregnancies', y='DiabetesPedigreeFunction',data=df, hue='Outcome', palette='Greens')
ax2= sns.scatterplot(x= user_data['pregnancies'], y=user_data['dpf'], s=150,color=color)
plt.xticks(np.arange(0, 20, 2))
plt.yticks(np.arange(0, 3, 0.5))
plt.title('0- Healthy & 1- Diabetic')
st.pyplot(fig_pregdpf)

# Glucose vs BloodPressure
st.header('Glucose vs BloodPressure Graph (Other Vs Yours)')
fig_gbp= plt.figure()
ax1= sns.scatterplot(x='Glucose', y='BloodPressure',data=df, hue='Outcome', palette='Greens')
ax2= sns.scatterplot(x= user_data['glucose'], y=user_data['bp'], s=150,color=color)
plt.xticks(np.arange(0, 200, 20))
plt.yticks(np.arange(0, 200, 20))
plt.title('0- Healthy & 1- Diabetic')
st.pyplot(fig_gbp)

# Glucose vs SkinThickness
st.header('Glucose vs SkinThickness Graph (Other Vs Yours)')
fig_gsk= plt.figure()
ax1= sns.scatterplot(x='Glucose', y='SkinThickness',data=df, hue='Outcome', palette='Greens')
ax2= sns.scatterplot(x= user_data['glucose'], y=user_data['sk'], s=150,color=color)
plt.xticks(np.arange(0, 200, 20))
plt.yticks(np.arange(0, 90, 8))
plt.title('0- Healthy & 1- Diabetic')
st.pyplot(fig_gsk)

# Glucose vs Insulin
st.header('Glucose vs Insulin Graph (Other Vs Yours)')
fig_gins= plt.figure()
ax1= sns.scatterplot(x='Glucose', y='Insulin',data=df, hue='Outcome', palette='Greens')
ax2= sns.scatterplot(x= user_data['glucose'], y=user_data['insulin'], s=150,color=color)
plt.xticks(np.arange(0, 200, 20))
plt.yticks(np.arange(0, 800, 55))
plt.title('0- Healthy & 1- Diabetic')
st.pyplot(fig_gins)

# Glucose vs BMI
st.header('Glucose vs BMI Graph (Other Vs Yours)')
fig_gbmi= plt.figure()
ax1= sns.scatterplot(x='Glucose', y='BMI',data=df, hue='Outcome', palette='Greens')
ax2= sns.scatterplot(x= user_data['glucose'], y=user_data['bmi'], s=150,color=color)
plt.xticks(np.arange(0, 200, 20))
plt.yticks(np.arange(0, 65, 5))
plt.title('0- Healthy & 1- Diabetic')
st.pyplot(fig_gbmi)

# Glucose vs DiabetesPedigreeFunction
st.header('Glucose vs DiabetesPedigreeFunction Graph (Other Vs Yours)')
fig_gdpf= plt.figure()
ax1= sns.scatterplot(x='Glucose', y='DiabetesPedigreeFunction',data=df, hue='Outcome', palette='Greens')
ax2= sns.scatterplot(x= user_data['glucose'], y=user_data['dpf'], s=150,color=color)
plt.xticks(np.arange(0, 200, 20))
plt.yticks(np.arange(0, 3, 0.5))
plt.title('0- Healthy & 1- Diabetic')
st.pyplot(fig_gdpf)



# output
st.header("Your Report: ")
output = ''
if user_result[0] == 0:
    output = "You are Healthy"
    st.balloons()
else:
    output = "Metha Kam Kahoo"
    st.warning("Sugar, Sugar, Sugar")
st.title(output)
st.header("Accuracy Score: ")
st.write(accuracy_score(y_test, rc.predict(x_test)))

# Classifiers scores
st.header("Classifiers Scores: ")
st.write("Precision Score: ", precision_score(y_test, rc.predict(x_test)))
st.write("Recall Score: ", recall_score(y_test, rc.predict(x_test)))
st.write("F1 Score: ", f1_score(y_test, rc.predict(x_test)))

# Confusion Matrix
st.header("Confusion Matrix: ")
cm = confusion_matrix(y_test, rc.predict(x_test))
cm
fig, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
st.pyplot(fig)






 




    








