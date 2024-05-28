import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import logging
from scipy.integrate import simpson
from scipy.optimize import curve_fit, OptimizeWarning
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the folder path containing the temperature data files
folder_path = '../data/tempcurves'

# Generate a list of files starting with 'PKK' in the specified folder
uploaded_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.xlsx')]
logging.info(f'Found {len(uploaded_files)} files')

# Load recurrence data from the specified path
recurrence_file_path = '../data/recurrence/recurrence.xlsx'
recurrence_data = pd.read_excel(recurrence_file_path)

# Verify the integrity of the recurrence_data DataFrame
print("Recurrence DataFrame structure:")
print(recurrence_data.info())
print("Recurrence DataFrame head:")
print(recurrence_data.head())
print("Number of unique patients in recurrence data:", recurrence_data['ID'].nunique())
print("Total number of recurrences:", recurrence_data['Recurrence (1 = yes; 0 = no)'].sum())

# Function to fit an exponential function
def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c

# Likelihood calculation function
def calculate_likelihood(temp_curve_df):
    # This is a placeholder for the likelihood calculation
    # Replace with the actual likelihood calculation logic
    return np.random.random()

# Feature extraction function for a single temperature curve
def extract_features(temp_curve_df):
    features = {}
    features['mean_temp'] = temp_curve_df['Temperature'].mean()
    features['min_temp'] = temp_curve_df['Temperature'].min()
    features['max_temp'] = temp_curve_df['Temperature'].max()
    features['std_temp'] = temp_curve_df['Temperature'].std()

    # Minimum temperature between 50-150 seconds
    temp_50_150 = temp_curve_df[(temp_curve_df['Time'] >= 50) & (temp_curve_df['Time'] <= 150)]
    features['min_temp_50_150'] = temp_50_150['Temperature'].min()

    # Slope of temperature decrease starting with timepoint 0
    if len(temp_curve_df) > 1:  # Ensure there are enough points to fit a line
        try:
            initial_slope = np.polyfit(temp_curve_df['Time'], temp_curve_df['Temperature'], 1)[0]
        except np.linalg.LinAlgError:
            initial_slope = np.nan  # Handle cases with fitting issues
    else:
        initial_slope = np.nan  # Handle cases with insufficient data points
    features['initial_slope'] = initial_slope

    # Slope of temperature increase between 150-180 seconds
    temp_150_180 = temp_curve_df[(temp_curve_df['Time'] >= 150) & (temp_curve_df['Time'] <= 180)]
    if len(temp_150_180) > 1:  # Ensure there are enough points to fit a line
        try:
            slope_150_180 = np.polyfit(temp_150_180['Time'], temp_150_180['Temperature'], 1)[0]
        except np.linalg.LinAlgError:
            slope_150_180 = np.nan  # Handle cases with fitting issues
    else:
        slope_150_180 = np.nan  # Handle cases with insufficient data points
    features['slope_150_180'] = slope_150_180

    # Cooling energy (integral below the temperature curve)
    cooling_energy = simpson(y=temp_curve_df['Temperature'], x=temp_curve_df['Time'])
    features['cooling_energy'] = cooling_energy

    # Mean quadratic error between temperature curve and exponential function
    if len(temp_curve_df) > 3:  # Ensure there are enough points to fit an exponential function
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                popt, _ = curve_fit(exp_func, temp_curve_df['Time'], temp_curve_df['Temperature'], maxfev=10000)
            fitted_curve = exp_func(temp_curve_df['Time'], *popt)
            mse = np.mean((temp_curve_df['Temperature'] - fitted_curve) ** 2)
        except (RuntimeError, OptimizeWarning, TypeError):
            mse = np.nan  # Handle cases where curve fitting fails
    else:
        mse = np.nan  # Handle cases with insufficient data points
    features['mean_quad_error_exp'] = mse

    # Add likelihood as a feature
    features['likelihood'] = calculate_likelihood(temp_curve_df)

    return features

# Initialize a DataFrame to store extracted features for each patient
extracted_features_list = []

# Process each temperature curve file
num_temp_curves = 0
for file in uploaded_files:
    patient_id = os.path.splitext(os.path.basename(file))[0]
    xls = pd.ExcelFile(file)
    for sheet_name in xls.sheet_names:
        temp_curve_df = pd.read_excel(xls, sheet_name=sheet_name, usecols=[0, 1])  # Only load the first two columns
        if temp_curve_df.shape[1] >= 2:  # Ensure there are at least 2 columns
            temp_curve_df.columns = ['Time', 'Temperature']  # Ensure columns are named correctly
            temp_curve_df = temp_curve_df[temp_curve_df['Time'] <= 180]  # Limit to first 180 seconds
            if temp_curve_df['Time'].iloc[-1] < 30:  # Filter out temperature curves fewer than 30 seconds
                continue
            features = extract_features(temp_curve_df)
            features['ID'] = patient_id
            extracted_features_list.append(features)
            num_temp_curves += 1
        else:
            logging.warning(f"File {file}, sheet {sheet_name} does not have at least 2 columns.")

extracted_features = pd.DataFrame(extracted_features_list)

# Verify the structure of the extracted features DataFrame
print("Extracted Features DataFrame structure:")
print(extracted_features.info())
print("Extracted Features DataFrame head:")
print(extracted_features.head())
print("Number of unique patients in extracted features:", extracted_features['ID'].nunique())
print("Total number of temperature curves:", num_temp_curves)
print("Mean number of temperature curves per patient:", num_temp_curves / extracted_features['ID'].nunique())

# Group extracted features by patient ID and calculate mean for each patient
extracted_features_grouped = extracted_features.groupby('ID').mean().reset_index()

# Merge extracted features with recurrence data
recurrence_data_with_features = recurrence_data.merge(extracted_features_grouped, on='ID', how='left')

# Verify the structure after merging
print("Merged DataFrame structure:")
print(recurrence_data_with_features.info())
print("Merged DataFrame head:")
print(recurrence_data_with_features.head())
print("Number of unique patients in merged data:", recurrence_data_with_features['ID'].nunique())

# Normalize the reconduction site value
recurrence_data_with_features['Normalized_Reconduction_Site'] = recurrence_data_with_features['Reconduction_Site'] / 4.0

# Prepare the dataset for training
# Replace missing values with the median value for numeric columns only
numeric_columns = recurrence_data_with_features.select_dtypes(include=[np.number]).columns
recurrence_data_with_features[numeric_columns] = recurrence_data_with_features[numeric_columns].fillna(recurrence_data_with_features[numeric_columns].median())

# Check for and handle infinite or excessively large values
# Replace infinities with NaN and then fill NaNs with the median value
recurrence_data_with_features.replace([np.inf, -np.inf], np.nan, inplace=True)

# Clip values that are excessively large
max_float = np.finfo(np.float32).max
recurrence_data_with_features[numeric_columns] = recurrence_data_with_features[numeric_columns].clip(upper=max_float)

recurrence_data_with_features[numeric_columns] = recurrence_data_with_features[numeric_columns].fillna(recurrence_data_with_features[numeric_columns].median())

# Add specified parameters to the model
specified_parameters = ['Sex (m=0 / f=1)', 'Height', 'Weight', 'Age', 'Time_to_recurrence', 'LV_EF', 'LA_FLAECHE', 'KREATININ', 'initial_rhythm (1=SR, 2=AF)']
feature_columns = list(recurrence_data_with_features.columns.difference(['Recurrence (1 = yes; 0 = no)', 'ID', 'Reconduction_Site', 'Normalized_Reconduction_Site']))
feature_columns += specified_parameters

# Ensure that feature columns are in X before training
X = recurrence_data_with_features[feature_columns]
y = recurrence_data_with_features['Normalized_Reconduction_Site']

# Debugging steps: Print useful information
print("Number of recurrences:", recurrence_data_with_features['Recurrence (1 = yes; 0 = no)'].sum())
print("Number of male patients:", (recurrence_data_with_features['Sex (m=0 / f=1)'] == 0).sum())
print("Number of female patients:", (recurrence_data_with_features['Sex (m=0 / f=1)'] == 1).sum())

# Summary statistics for numeric features
print("Summary statistics for numeric features:")
print(recurrence_data_with_features[numeric_columns].describe().transpose())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Log the results
logging.info(f'Mean Squared Error: {mse}')
logging.info(f'R2 Score: {r2}')

# Additional debugging information
print("Training set size:", X_train.shape[0])
print("Testing set size:", X_test.shape[0])
print("Training R2:", model.score(X_train, y_train))
print("Testing R2:", r2)

# Plot true vs. predicted values
plt.figure()
plt.scatter(y_test, y_pred, color='darkorange', label='Predicted vs. True')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs. Predicted Values')
plt.legend(loc='upper left')
plt.show()

# Plot histogram of residuals
residuals = y_test - y_pred
plt.figure()
plt.hist(residuals, bins=20, color='gray', edgecolor='black')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
plt.show()

# Visualize a single tree from the forest
plt.figure(figsize=(20, 10))
plot_tree(model.estimators_[0], filled=True, feature_names=feature_columns, rounded=True, precision=1)
plt.title("Visualization of a Single Tree from the RandomForestRegressor")
plt.show()
