#Import the streamlit package
import streamlit as st

#Instantiate a streamlit app
##Set title to Fetal monitoring project

st.title(':baby: Fetal monitoring project: :baby:')
st.write('A fetal monitoring is critical for good neonatal outcomes in labor and delivery')
st.write('Here we analyze data from fetal cardiac monioring using a cardiotograph')


#Load data processing libraries
import pandas as pd
import numpy as np

#Read data
fetal = pd.read_csv('fetal_dataset.csv')
st.write(fetal.sample(6))

#Explore the data
st.write('Summary statistics')
st.write(fetal.describe().T)

#Data analysis and exploration
st.title('Data analysis and exploration')
st.write('Check for class imbalance on the target variable')

"""Features"""

"""'baseline value' FHR baseline (beats per minute)"""
"""'accelerations' Number of accelerations per second """
"""fetal_movement: Number of fetal movements per second """
""" uterine_contractions: Number of uterine contractions per second """
"""light_decelerations: Number of light decelerations per second """
"""severe_decelerations: Number of severe decelerations per second"""
"""prolongued_decelerations: Number of prolonged decelerations per second """
"""abnormal_short_term_variability: Percentage of time with abnormal short term variability"""
"""mean_value_of_short_term_variability: Mean value of short term variability"""
"""percentage_of_time_with_abnormal_long_term_variability: Percentage of\
      time with abnormal long term variability """
"""mean_value_of_long_term_variability: Mean value of long term variability"""
"""histogram_width: Width of FHR histogram"""
"""histogram_min: Minimum (low frequency) of FHR histogram"""
"""histogram_max: Maximum (high frequency) of FHR histogram"""
"""histogram_number_of_peaks: Number of histogram peaks"""
"""histogram_number_of_zeroes: Number of histogram zeros """
"""histogram_mode: Histogram mode """
"""histogram_mean: Histogram mean """
"""histogram_median: Histogram median """
"""histogram_variance: Histogram variance """
"""histogram_tendency: Histogram tendency"""

"""Target"""

"""'fetal_health' Tagged as 1 (Normal), 2 (Suspect) and 3 (Pathological)"""
import os

# Use a non-interactive backend for matplotlib
if "DYNO" in os.environ:
    import matplotlib
    matplotlib.use("agg")
    
if "DYNO" in os.environ:
      import seaborn
      seaborn.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler

# Create a count plot using Plotly Express
fig = px.histogram(fetal, x="fetal_health", color ="fetal_health", category_orders={"fetal_health": [1, 2, 3]})

# Set layout properties
fig.update_layout(
    xaxis_title='Fetal Health',
    yaxis_title='Count',
    title='Distribution of Fetal Health',
    showlegend= True
)

# Display the plot in Streamlit
st.plotly_chart(fig)
st.write('The figure above shows that there is a class imbalance on fetal outcomes where\
         1=Normal, 2=Suspect, and 3=Pathological')


# Calculate the correlation matrix
corrmat = fetal.corr()

# Create a Plotly heatmap
heatmap_trace = go.Heatmap(
    z=corrmat.values,
    x=corrmat.columns,
    y=corrmat.columns,
    colorscale='RdBu',
    zmin=-1,
    zmax=1
)

layout = go.Layout(
    title='Correlation Matrix Heatmap',
    xaxis=dict(title='Features'),
    yaxis=dict(title='Features')
)

fig = go.Figure(data=[heatmap_trace], layout=layout)

# Display the heatmap in Streamlit
st.plotly_chart(fig)

st.write(' "accelerations","prolongued_decelerations", \
         "abnormal_short_term_variability", \
         "percentage_of_time_with_abnormal_long_term_variability" \
         and "mean_value_of_long_term_variability" \
         are the features with higher correlation with fetal_health.')


st.title('RandomForest Classfier')
# Define features and target
features = ['accelerations', 'prolongued_decelerations', 'abnormal_short_term_variability', 'percentage_of_time_with_abnormal_long_term_variability', 'mean_value_of_long_term_variability']
target = 'fetal_health'

# Split the data into training and testing sets
X = fetal[features]
y = fetal[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
clf = RandomForestClassifier(random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluate the model
st.write('Confusion matrix')
st.write("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
st.write('Classification report')
st.write("\nClassification Report:\n", classification_report(y_test, y_pred))
"""the model is performing reasonably well. It has high precision and recall for class 1, \
    meaning it's good at identifying instances of class \
        1. Class 2 has slightly lower precision and recall, indicating some misclassifications. \
            Class 3 also has good precision and recall. \
                The weighted average F1-score of 0.93 suggests that the model is providing \
                    a good overall balance between precision and recall across all classes."""


#Correct for class imbalance
# Split the data into training and testing sets
X = fetal[features]
y = fetal[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply under-sampling to balance the classes
rus = RandomUnderSampler(random_state=42)
X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

# Initialize the Random Forest Classifier
clf = RandomForestClassifier(random_state=42)

# Train the classifier on the resampled data
clf.fit(X_train_resampled, y_train_resampled)

# Predictions
y_pred = clf.predict(X_test)

# Evaluate the model
classification_rep = classification_report(y_test, y_pred, target_names=['Normal', 'Suspect', 'Pathological'])

# Display classification report in Streamlit
st.text("Classification Report:\n" + classification_rep)
