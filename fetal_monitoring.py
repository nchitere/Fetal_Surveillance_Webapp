#Import the streamlit package
import streamlit as st

#Instantiate a streamlit app
##Set title to Fetal monitoring project

st.title(':baby: Fetal monitoring project: :baby:')
st.write('The monitoring of fetal well-being stands as a pivotal factor in shaping positive neonatal outcomes in pregnancy, labor and delivery.')
st.write('Knowing which indicators to focus on can be the difference between good vs bad neonatal outcomes.')
st.write('In this project we use machine learning to singleout which features are the best predictors of fetal outcomes.')
st.write('Data source: Kaggle')
st.write('This dataset contains 2126 records of features extracted from Cardiotocogram exams,\
which were then classified by three expert obstetritians into 3 classes:1. Normal, 2.Suspect, & 3.Pathological')

#Load data processing libraries
import pandas as pd
import numpy as np

#Load data visualization libaries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go

#Load machine learning libaries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier


#Read data
fetal = pd.read_csv('fetal_dataset.csv') 
"""Here is a snippet of the fetal surveillance data set"""
st.write(fetal.sample(6))
st.write(fetal.shape)

#Features and Target
st.title('Features')

"""1. baseline value: FHR baseline (beats per minute)"""
"""2. accelerations: Number of accelerations per second """
"""3. fetal_movement: Number of fetal movements per second """
"""4. uterine_contractions: Number of uterine contractions per second """
"""5. light_decelerations: Number of light decelerations per second """
"""6. severe_decelerations: Number of severe decelerations per second"""
"""7. prolongued_decelerations: Number of prolonged decelerations per second """
"""8. abnormal_short_term_variability: Percentage of time with abnormal short term variability"""
"""9. mean_value_of_short_term_variability: Mean value of short term variability"""
"""10. percentage_of_time_with_abnormal_long_term_variability: Percentage of\
      time with abnormal long term variability """
"""11. mean_value_of_long_term_variability: Mean value of long term variability"""
"""12. histogram_width: Width of FHR histogram"""
"""13. histogram_min: Minimum (low frequency) of FHR histogram"""
"""14. histogram_max: Maximum (high frequency) of FHR histogram"""
"""15. histogram_number_of_peaks: Number of histogram peaks"""
"""16. histogram_number_of_zeroes: Number of histogram zeros """
"""17. histogram_mode: Histogram mode """
"""18. histogram_mean: Histogram mean """
"""19. histogram_median: Histogram median """
"""20. histogram_variance: Histogram variance """
"""21. histogram_tendency: Histogram tendency"""

st.title('Target')

"""1. fetal_health: Tagged as 1 (Normal), 2 (Suspect) and 3 (Pathological)"""

#Explore the summary statistics
st.write('Summary statistics')
st.write(fetal.describe().T)

#Data analysis and exploration
st.title('Data analysis and exploration')
st.write('Data missingness check')

# Calculate missing data percentages for each column
missing_percentages = (fetal.isnull().sum() / len(fetal)) * 100

# Sort columns by missing percentage in descending order
missing_percentages = missing_percentages.sort_values(ascending=False)

# Display missing data percentages in a table
st.write("Missing Data Percentages:")
st.write(missing_percentages)
st.write('As shown in the above summary table, there is no missing data')
st.write('Since machine learning models such as RandomForest Classifiers cannot handle missing data, \
and no data is missing, no further is required.')


st.title('Checking for class imbalance on the target variable')
st.write('Class imbalance in the target can result in a biased model,\
might not generalize well on test data sets \
and since the pathological neonatal outcomes are not the norm, yet flagging at risk neonates  is critical, \
it is important to check for and correct for class imbalance where appicable')

# Create a count plot of the target column (fetal_health) using Plotly Express
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
st.write('The displayed output indicates that majority of the monitored fetuses are categorized normal fetal health,\
while a smaller portion falls into the suspect or pathological categories.')

st.title('Feature selection')
# Calculate the correlation matrix
corrmat = fetal.corr()

      
# Create a Plotly heatmap from corrmat
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

st.write('The features with the highest correlation to fetal health, as per the correlation matrix, are:'
         '\n 1. Accelerations'
         '\n 2. Prolonged Decelerations'
         '\n 3. Abnormal Short-term Variability'
         '\n 4. Percentage of Time with Abnormal Long-term Variability'
         '\n 5. Mean Value of Long-term Variability')


st.title('Models')
st.title('RandomForest Classfier')
st.write('The Random Forest Classifier demonstrates its robustness in fetal health prediction, harnessing a diverse array of features for precise evaluations.'
         '\nThe following confusion matrix offers a comprehensive view of classification outcomes,'
         '\nwhile the accompanying classification report presents valuable metrics such as precision, recall, and F1-score for each class.')

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

st.title('Model evaluation on uncorrected class imbalance')

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


st.write('Correcting for class imbalance in the target variable(fetal_health)')

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


st.title('Model evaluation after accounting for class imbalance')

classification_rep = classification_report(y_test, y_pred, target_names=['Normal', 'Suspect', 'Pathological'])

# Display classification report in Streamlit
st.text("Classification Report:\n" + classification_rep)

st.write("The classification report evaluates the model's performance in predicting fetal health outcomes categorized as 'Normal,' 'Suspect,' and 'Pathological.'\n")

st.write("For the 'Normal' class, the model demonstrates high precision (97%) and moderate recall (81%), resulting in an F1-score of 0.88.\n")

st.write("In the 'Suspect' class, precision is lower (57%), but recall is higher (86%), yielding an F1-score of 0.69.\n")

st.write("For the 'Pathological' class, precision is modest (53%), while recall is high (97%), leading to an F1-score of 0.68.\n")

st.write("Overall accuracy is 83%, and macro averages indicate precision-recall balance across classes (69% precision, 88% recall, 75% F1-score). Weighted averages consider class frequencies, resulting in balanced performance metrics (88% precision, 83% recall, 84% F1-score).\n")

#st.write("The report offers insights into the model's competence across fetal health categories, revealing strengths and areas for enhancement in predicting different cases.")

st.title('Reciever operating characteristics')

st.set_option('deprecation.showPyplotGlobalUse', False)
from sklearn.preprocessing import label_binarize

# Binarize the true labels for multiclass ROC analysis
y_test_binarized = label_binarize(y_test, classes=[1, 2, 3])
n_classes = y_test_binarized.shape[1]
y_pred_probs = clf.predict_proba(X_test)
# Calculate ROC AUC scores for each class
roc_auc_scores = roc_auc_score(y_test_binarized, y_pred_probs, average=None)

# Plot ROC curves for each class
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_probs[:, i])
    plt.plot(fpr, tpr, label=f'Class {target[i]} (AUC = {roc_auc_scores[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()
st.pyplot()

# st.title('Xgboost model')
# # Instantiate an XGBoost model
# #xgb = XGBClassifier(random_state=42)
# class_labels = [1, 2, 3]
# xgb_classifier = XGBClassifier(objective='multi:softmax', num_class=len(class_labels), classes=class_labels)
# # Train the model
# xgb.fit(X_train_resampled, y_train_resampled)

# # Make predictions
# y_pred = xgb.predict(X_test)

# # Display model evaluation results
# st.write("Model evaluation:")
# classification_rep = classification_report(y_test, y_pred)
# st.write(classification_rep)


