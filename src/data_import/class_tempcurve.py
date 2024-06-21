import numpy as np

class class_tempcurve: 
        
    # initiates a new instance of a tempcurve with the resprective raw.data 
    def __init__(self, temp_curve_df, patient_id, patient_vein_count): 
        
        self.raw_data = temp_curve_df    
        self.patient_id = patient_id
        self.trace_id = patient_id + str(patient_vein_count)
        
        self.cutted_trace = []
        self.features = {}
    
    
    # find the dipping point of a temp curve and saves a cutted_trace, that starts with the dipping point 
    def find_dipping_point(self):
        
        point_20 =self.raw_data[self.raw_data.Temperature <= 20].index[0]  
        
        dipping_point = 0
        
        for i in range(point_20,0, -1): 
            if self.raw_data['Temperature'].iloc[i]>= self.raw_data['Temperature'].iloc[i - 1]:
                dipping_point = i
                break
        
        self.cutted_trace = self.raw_data.iloc[dipping_point:]  
        self.cutted_trace.iloc[:,0]= range(1,len(self.cutted_trace['Time'])+1)
        
        
    def extract_features(self):
        
        self.features['mean_temp'] = self.cutted_trace['Temperature'].mean()
        self.features['min_temp'] = self.cutted_trace['Temperature'].min()
        self.features['max_temp'] = self.cutted_trace['Temperature'].max()
        self.features['std_temp'] = self.cutted_trace['Temperature'].std()
        self.features['length'] = self.cutted_trace['Temperature'].shape[0]
        
        # kinetics of decline
        try: 
            self.features['t30']=self.cutted_trace[self.cutted_trace.Temperature <= 30].iloc[0,0]
            self.features['t20']=self.cutted_trace[self.cutted_trace.Temperature <= 20].iloc[0,0]
            self.features['t10']=self.cutted_trace[self.cutted_trace.Temperature <= 10].iloc[0,0]
            self.features['t0']=self.cutted_trace[self.cutted_trace.Temperature <= 0].iloc[0,0]
            self.features['t_10']=self.cutted_trace[self.cutted_trace.Temperature <= -10].iloc[0,0]
            self.features['t_20']=self.cutted_trace[self.cutted_trace.Temperature <= -20].iloc[0,0]
            self.features['t_30']=self.cutted_trace[self.cutted_trace.Temperature <= -30].iloc[0,0]
            self.features['t_40']=self.cutted_trace[self.cutted_trace.Temperature <= -40].iloc[0,0]
            
        except: 
            pass
        
        # Slope of temperature decrease starting with timepoint 0
        try:
                initial_slope = np.polyfit(self.cutted_trace['Time'], self.cutted_trace['Temperature'], 1)[0]
        except np.linalg.LinAlgError:
                initial_slope = np.nan  # Handle cases with fitting issues
        
        self.features['initial_slope'] = initial_slope
        
        # Cooling energy (integral below the maximum temperature)
        self.features['cooling_energy'] = sum(x - self.features['max_temp'] for x in self.cutted_trace['Temperature'])
        self.features['average_cooling_energy'] = self.features['cooling_energy']/ self.features['length']
        