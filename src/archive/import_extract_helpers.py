import numpy as np
from scipy.integrate import simpson

# check wheter tempcurves fulfills all inclusion criteria
def is_valid_tempcurve(temp_curve_df): 
    
    if temp_curve_df['Time'].iloc[-1] < 100:    # filter by length of temperaturecurve
        return False   
    
    if min(temp_curve_df['Temperature'])>0:     # filter curves that never actually cool down
        return False
    
    return True 

# cut temp curve at dipping point
def find_dipping_point(temp_curve_df):    
    point_20 =temp_curve_df[temp_curve_df.Temperature <= 20].index[0]  
    
    dipping_point = 0
    
    for i in range(point_20,0, -1): 
        if temp_curve_df['Temperature'].iloc[i]>= temp_curve_df['Temperature'].iloc[i - 1]:
            dipping_point = i
            break
    
    output = temp_curve_df.iloc[dipping_point:]  
    output.iloc[:,0]= range(1,len(output['Time'])+1)
    
    return output
    
# Feature extraction function for a single temperature curve
def extract_features(temp_curve_df):
    features = {}
    features['mean_temp'] = temp_curve_df['Temperature'].mean()
    features['min_temp'] = temp_curve_df['Temperature'].min()
    features['max_temp'] = temp_curve_df['Temperature'].max()
    features['std_temp'] = temp_curve_df['Temperature'].std()
    features['length'] = temp_curve_df['Temperature'].shape[0]
    
    # kinetics of decline
    try: 
        features['t30']=temp_curve_df[temp_curve_df.Temperature <= 30].iloc[0,0]
        features['t20']=temp_curve_df[temp_curve_df.Temperature <= 20].iloc[0,0]
        features['t10']=temp_curve_df[temp_curve_df.Temperature <= 10].iloc[0,0]
        features['t0']=temp_curve_df[temp_curve_df.Temperature <= 0].iloc[0,0]
        features['t_10']=temp_curve_df[temp_curve_df.Temperature <= -10].iloc[0,0]
        features['t_20']=temp_curve_df[temp_curve_df.Temperature <= -20].iloc[0,0]
        features['t_30']=temp_curve_df[temp_curve_df.Temperature <= -30].iloc[0,0]
        features['t_40']=temp_curve_df[temp_curve_df.Temperature <= -40].iloc[0,0]
        
    except: 
        pass
    
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

    # Cooling energy (integral below the temperature curve)
    cooling_energy = simpson(y=(temp_curve_df['Temperature']), x=temp_curve_df['Time'])
    features['cooling_energy'] = cooling_energy

    return features
    
    

