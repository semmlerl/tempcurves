import numpy as np
from scipy.integrate import simpson
from scipy.optimize import curve_fit, OptimizeWarning
import warnings


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
    features['length'] = temp_curve_df['Temperature'].shape[0]

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