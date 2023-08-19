# Fetal Surveillance Web App

Fetal monitoring plays a crucial role in ensuring favorable neonatal outcomes during labor and delivery. This repository focuses on the analysis of fetal cardiac monitoring data utilizing a cardiotocograph.

The primary objective of this project is to employ machine learning techniques for predicting neonatal outcomes, categorized into three classes: 1 (Normal), 2 (Suspect), and 3 (Pathological).

The chosen model for this prediction task is the RandomForest Classifier, which effectively identifies the features that best contribute to predicting the target outcomes.

# Targets and Features

## Features
baseline value (FHR baseline): Heart rate baseline measured in beats per minute.
accelerations: Number of accelerations per second.
fetal_movement: Frequency of fetal movements per second.
uterine_contractions: Rate of uterine contractions per second.
light_decelerations: Count of light decelerations per second.
severe_decelerations: Count of severe decelerations per second.
prolongued_decelerations: Count of prolonged decelerations per second.
abnormal_short_term_variability: Percentage of time exhibiting abnormal short-term variability.
mean_value_of_short_term_variability: Average value of short-term variability.
percentage_of_time_with_abnormal_long_term_variability: Percentage of time displaying abnormal long-term variability.
mean_value_of_long_term_variability: Average value of long-term variability.
histogram_width: Width of the FHR histogram.
histogram_min: Minimum (low frequency) of the FHR histogram.
histogram_max: Maximum (high frequency) of the FHR histogram.
histogram_number_of_peaks: Count of peaks in the histogram.
histogram_number_of_zeroes: Count of zeroes in the histogram.
histogram_mode: Mode of the histogram.
histogram_mean: Mean value of the histogram.
histogram_median: Median value of the histogram.
histogram_variance: Variance of the histogram.
histogram_tendency: Tendency of the histogram.

# Target
The target variable is labeled as 'fetal_health' and categorized into three classes:

1 (Normal): Normal fetal health.
2 (Suspect): Suspected deviation from normal fetal health.
3 (Pathological): Pathological fetal health.
Feel free to explore the code and contribute to enhancing the accuracy and effectiveness of the fetal health prediction using this web application. Your contributions are highly valued!

If you encounter any issues, have suggestions, or would like to collaborate, please open an issue or pull request.

Let's work together to improve neonatal outcomes through advanced fetal surveillance techniques!
