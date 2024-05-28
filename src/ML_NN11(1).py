import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, precision_recall_curve, confusion_matrix, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis
from scipy.integrate import trapezoid
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import l1, l2
from scikeras.wrappers import KerasClassifier
from imblearn.over_sampling import ADASYN
from tensorflow.keras.utils import plot_model
import time

# Use tensorflow-directml if available
try:
    import tensorflow_directml as tf
    print("Using tensorflow-directml")
except ImportError:
    import tensorflow as tf
    print("Using standard tensorflow")

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to estimate the runtime of feature extraction
def estimate_runtime():
    start_time = time.time()
    sample_file = uploaded_files[0]
    excel_file = pd.ExcelFile(sample_file)
    sheet_data = excel_file.parse(excel_file.sheet_names[0])
    extract_features(sheet_data, 'sample', excel_file.sheet_names[0])
    end_time = time.time()
    single_run_time = end_time - start_time
    estimated_total_time = single_run_time * len(uploaded_files) * len(excel_file.sheet_names)
    return estimated_total_time

# Define the folder path containing the temperature data files
folder_path = r'C:\\Users\\hohen\\Documents\\Forschung\\Machine learning\\Kryoablation - Prädiktion Rezidiv nach Temperaturkurve\\Test\\temperature curves'

# Generate a list of files starting with 'PKK' in the specified folder
uploaded_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.startswith('PKK') and f.endswith('.xlsx')]
logging.info(f'Found {len(uploaded_files)} files starting with "PKK"')

# Load recurrence data
recurrence_file_path = os.path.join(folder_path, 'database recurrence.xlsx')
recurrence_data = pd.read_excel(recurrence_file_path)
recurrence_data.columns = ['ID', 'Recurrence', 'ReconductionSite', 'Sex', 'Height', 'Weight', 'Age', 'TimeToRecurrence', 'LVEF', 'LA_Size', 'Creatinine', 'InitialRhythm']
recurrence_data['Recurrence'] = recurrence_data['Recurrence'].fillna(0).astype(int)
recurrence_data['ID'] = recurrence_data['ID'].astype(str)  # Ensure ID is of type string
logging.info(f'Recurrence data loaded with {recurrence_data.shape[0]} entries')

# Function to extract features from temperature curves using 'col1' as the temperature column, considering only the first 180 seconds
def extract_features(sheet_data, file_id, sheet_name):
    # Limit the data to the first 180 seconds
    sheet_data_180 = sheet_data[sheet_data['Unnamed: 0'] <= 180]
    
    # Filter out temperature curves that are fewer than 30 seconds
    if sheet_data_180['Unnamed: 0'].iloc[-1] < 30:
        return None
    
    # Add small noise to avoid precision loss
    noise = np.random.normal(0, 1e-5, sheet_data_180['col1'].shape)
    temp_data_noisy = sheet_data_180['col1'] + noise
    
    # Convert to numpy array
    temp_data_noisy_np = temp_data_noisy.to_numpy()
    
    if len(temp_data_noisy_np) < 2:
        # If there are fewer than 2 data points, return NaNs for all features
        return {
            'ID': file_id,
            'Sheet': sheet_name,
            'MinTemp': np.nan,
            'TimeToMinTemp': np.nan,
            'NegativeSlope': np.nan,
            'MeanTemp50_150': np.nan,
            'TempStd': np.nan,
            'TempVariance': np.nan,
            'TempSkewness': np.nan,
            'TempKurtosis': np.nan,
            'FFTReal': np.nan,
            'FFTImag': np.nan,
            'FFTMag1': np.nan,
            'FFTMag2': np.nan,
            'FFTMag3': np.nan,
            'FFTMag4': np.nan,
            'FFTMag5': np.nan,
            'TempFluctuation': np.nan,
            'UpstrokeSlope': np.nan,
            'PrematureTermination': np.nan,
            'MinTemp50_150': np.nan,
            'SlopeDecrease0_150': np.nan,
            'SlopeIncrease150_180': np.nan,
            'CoolingEnergy': np.nan,
            'HistogramBelow': np.nan,
            'MeanQuadraticError': np.nan
        }
    
    # Calculate the minimal temperature and time to reach it
    min_temp = temp_data_noisy_np.min()
    time_to_min_temp = sheet_data_180[temp_data_noisy_np == min_temp]['Unnamed: 0'].min()
    
    # Calculate the negative slope (steepest negative slope)
    slopes = np.diff(temp_data_noisy_np) / np.diff(sheet_data_180['Unnamed: 0'].to_numpy())
    negative_slope = slopes.min() if len(slopes) > 0 else np.nan

    # Calculate mean temperature between 50-150 seconds
    temp_50_150 = temp_data_noisy_np[(sheet_data_180['Unnamed: 0'] >= 50) & (sheet_data_180['Unnamed: 0'] <= 150)]
    mean_temp_50_150 = temp_50_150.mean() if len(temp_50_150) > 0 else np.nan

    # Calculate the standard deviation of the temperature data (variability)
    temp_std = temp_data_noisy_np.std()

    # Calculate variance, skewness, and kurtosis
    temp_variance = temp_data_noisy_np.var()
    temp_skewness = skew(temp_data_noisy_np)
    temp_kurtosis = kurtosis(temp_data_noisy_np)

    # Fourier analysis: Compute the FFT of the temperature data
    yf = fft(temp_data_noisy_np)
    xf = fftfreq(len(sheet_data_180), 1)  # Assuming the time intervals are in seconds
    
    # Get the magnitudes of the first few Fourier coefficients
    fft_magnitudes = np.abs(yf)[:5]
    if len(fft_magnitudes) < 5:
        fft_magnitudes = np.pad(fft_magnitudes, (0, 5 - len(fft_magnitudes)), 'constant')
    
    # Additional Features
    # Consistency of temperature (small fluctuations)
    temp_diff = np.diff(temp_data_noisy_np)
    temp_fluctuation = np.std(temp_diff)
    
    # Upstroke speed (slope of temperature increase in the initial period)
    if len(temp_data_noisy_np) > 1:
        upstroke_slope = (temp_data_noisy_np[1] - temp_data_noisy_np[0]) / (sheet_data_180['Unnamed: 0'].iloc[1] - sheet_data_180['Unnamed: 0'].iloc[0])
    else:
        upstroke_slope = np.nan
    
    # Premature termination
    premature_termination = int(len(sheet_data_180) < 180)
    
    # Minimal temperature between 50-150 seconds
    min_temp_50_150 = np.min(temp_50_150) if len(temp_50_150) > 0 else np.nan
    
    # Slope of temperature decrease (0-150 seconds)
    if len(temp_data_noisy_np) > 150:
        slope_decrease_0_150 = (temp_data_noisy_np[150] - temp_data_noisy_np[0]) / (sheet_data_180['Unnamed: 0'].iloc[150] - sheet_data_180['Unnamed: 0'].iloc[0])
    else:
        slope_decrease_0_150 = np.nan
    
    # Slope of temperature increase (150-180 seconds)
    if len(temp_data_noisy_np) > 150:
        slope_increase_150_180 = (temp_data_noisy_np[-1] - temp_data_noisy_np[150]) / (sheet_data_180['Unnamed: 0'].iloc[-1] - sheet_data_180['Unnamed: 0'].iloc[150])
    else:
        slope_increase_150_180 = np.nan
    
    # Cooling energy (integral over time below the temperature curve)
    cooling_energy = trapezoid(temp_data_noisy_np, dx=1)
    
    # Time below certain temperature (histogram in 5°C steps)
    histogram_bins = np.histogram(temp_data_noisy_np, bins=np.arange(np.min(temp_data_noisy_np), np.max(temp_data_noisy_np) + 5, 5))
    
    # Mean quadratic error between temperature curve and exponential function
    valid_indices = temp_data_noisy_np > 0
    if np.any(valid_indices):
        exp_fit = np.polyfit(sheet_data_180['Unnamed: 0'][valid_indices], np.log(temp_data_noisy_np[valid_indices]), 1)
        exp_curve = np.exp(exp_fit[1] + exp_fit[0] * sheet_data_180['Unnamed: 0'][valid_indices])
        mean_quadratic_error = np.mean((temp_data_noisy_np[valid_indices] - exp_curve) ** 2)
    else:
        mean_quadratic_error = np.nan
    
    features = {
        'ID': file_id,
        'Sheet': sheet_name,
        'MinTemp': min_temp,
        'TimeToMinTemp': time_to_min_temp,
        'NegativeSlope': negative_slope,
        'MeanTemp50_150': mean_temp_50_150,
        'TempStd': temp_std,
        'TempVariance': temp_variance,
        'TempSkewness': temp_skewness,
        'TempKurtosis': temp_kurtosis,
        'FFTReal': yf.real[0],
        'FFTImag': yf.imag[0],
        'FFTMag1': fft_magnitudes[0],
        'FFTMag2': fft_magnitudes[1],
        'FFTMag3': fft_magnitudes[2],
        'FFTMag4': fft_magnitudes[3],
        'FFTMag5': fft_magnitudes[4],
        'TempFluctuation': temp_fluctuation,
        'UpstrokeSlope': upstroke_slope,
        'PrematureTermination': premature_termination,
        'MinTemp50_150': min_temp_50_150,
        'SlopeDecrease0_150': slope_decrease_0_150,
        'SlopeIncrease150_180': slope_increase_150_180,
        'CoolingEnergy': cooling_energy,
        'HistogramBelow': histogram_bins[0].sum(),
        'MeanQuadraticError': mean_quadratic_error
    }
    
    return features

# Estimate the runtime
estimated_total_time = estimate_runtime()
print(f'Estimated total runtime for feature extraction: {estimated_total_time / 60:.2f} minutes')

# Extract features for all temperature curves
all_features = []
total_temperature_curves = 0
for file_path in uploaded_files:
    excel_file = pd.ExcelFile(file_path)
    file_id = os.path.basename(file_path).split('.')[0]
    
    for sheet_name in excel_file.sheet_names:
        sheet_data = excel_file.parse(sheet_name)
        features = extract_features(sheet_data, file_id, sheet_name)
        if features is not None:  # Skip curves that do not meet the criteria
            all_features.append(features)
            total_temperature_curves += 1

logging.info(f'Extracted features for {len(all_features)} temperature curves')
print(f'Total number of temperature curves: {total_temperature_curves}')

# Create a DataFrame from the extracted features
features_df = pd.DataFrame(all_features)

# Merge features with recurrence data
data = pd.merge(features_df, recurrence_data, how='left', left_on='ID', right_on='ID')
logging.info(f'Merged features with recurrence data: {len(data)} total entries')

# Drop rows with missing target values
data = data.dropna(subset=['Recurrence'])

# Handle missing feature values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(data.drop(columns=['ID', 'Recurrence', 'Sheet']))
y = data['Recurrence']

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
logging.info(f'Data split into training and testing sets: {len(X_train)} train samples, {len(X_test)} test samples')

# Handle class imbalance using ADASYN
adasyn = ADASYN(random_state=42)
X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)
logging.info(f'Resampled training set: {len(X_train_resampled)} samples')

# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_resampled), y=y_train_resampled)

# Define a function to create the Keras model with additional regularization and tuning parameters
def create_model(neurons=128, dropout_rate=0.3, l2_lambda=0.001, optimizer='adam'):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(neurons, activation='relu', kernel_regularizer=l2(l2_lambda)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Wrap Keras model with scikit-learn
model = KerasClassifier(model=create_model, verbose=0)

# Define the grid search parameters
param_grid = {
    'model__neurons': [64, 128],
    'model__dropout_rate': [0.3, 0.5],
    'model__l2_lambda': [0.001, 0.01],
    'optimizer': ['adam', 'rmsprop'],
    'batch_size': [10, 20],
    'epochs': [50, 100]
}

# Perform grid search with cross-validation
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=StratifiedKFold(5), scoring='accuracy', verbose=1)
grid_result = grid.fit(X_train_resampled, y_train_resampled)

# Summarize the results
print(f'Best score: {grid_result.best_score_}')
print('Best parameters:')
for param, value in grid_result.best_params_.items():
    print(f'{param}: {value}')

# Evaluate the best model with the test set
best_model = grid_result.best_estimator_

y_pred_prob = best_model.predict_proba(X_test)[:, 1]  # Get the probability of the positive class
y_pred = (y_pred_prob > 0.5).astype(int)

# Classification report and confusion matrix
classification_report_res = classification_report(y_test, y_pred, zero_division=1)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Model accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Classification report:\n{classification_report_res}')
print(f'Confusion matrix:\n{conf_matrix}')

# ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()
logging.info('ROC curve plotted')

# Compute Precision-Recall curve
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_prob)
plt.figure()
plt.plot(recall, precision, marker='.', color='blue')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Find the best threshold
f1_scores = 2 * (precision * recall) / (precision + recall)
best_threshold_index = np.argmax(f1_scores)
best_threshold = thresholds_pr[best_threshold_index]

print(f'Best F1 score: {f1_scores[best_threshold_index]}')
print(f'Best threshold: {best_threshold}')

# Predict using the best threshold
y_pred_best_threshold = (y_pred_prob > best_threshold).astype(int)
accuracy_best_threshold = accuracy_score(y_test, y_pred_best_threshold)
classification_report_best_threshold = classification_report(y_test, y_pred_best_threshold, zero_division=1)
conf_matrix_best_threshold = confusion_matrix(y_test, y_pred_best_threshold)

print(f'Accuracy with best threshold: {accuracy_best_threshold}')
print(f'Classification report with best threshold:\n{classification_report_best_threshold}')
print(f'Confusion matrix with best threshold:\n{conf_matrix_best_threshold}')

# Visualize the neural network architecture
plot_model(best_model.model_, to_file='neural_network_architecture.png', show_shapes=True, show_layer_names=True)
plt.figure()
img = plt.imread('neural_network_architecture.png')
plt.imshow(img)
plt.axis('off')
plt.show()
