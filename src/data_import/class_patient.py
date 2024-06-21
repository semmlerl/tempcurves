import pandas as pd
from class_tempcurve import class_tempcurve

class class_patient: 
    
    def __init__(self, patient_id, xls): 
        
        ## initializes patient_id and tempcurve_list
        self.patient_id = patient_id 
        self.tempcurve_list = []
          
        # initialize a counter for the individual traces per patient 
        patient_vein_count = 0  
        
        ## loop through all sheets of the respective file 
        for sheet_name in xls.sheet_names:
            
            temp_curve_df = pd.read_excel(xls, sheet_name=sheet_name, usecols=[0, 1])  # Only load the first two columns
            temp_curve_df.columns = ['Time', 'Temperature']  # Ensure columns are named correctly
         
                  
            # checks wheter the respective file is valid 
            if is_valid_tempcurve(temp_curve_df):  
                
                # generates a new instance of the tempcurve class with the respective rawdata and the patient_id and trace_id
                tempcurve = class_tempcurve(temp_curve_df, patient_id, patient_vein_count)
                
                ##cutting the temperature curve at the dipping point to exclude the nonspecific "waiting time at the beginning"
                tempcurve.find_dipping_point()
                                           
                ## extract features and add them to feature_list
                tempcurve.extract_features()
                
                self.tempcurve_list.append(tempcurve)  
                
                patient_vein_count = patient_vein_count + 1
                        
        self.vein_count = len(self.tempcurve_list)

    def calculate_dipping_points(self): 
        
        for tempcurve in self.tempcurve_list: 
            tempcurve.find_dipping_point()
    
    def extract_features(self): 
        
        for tempcurve in self.tempcurve_list: 
            tempcurve.extract_features()
    
    def add_clinical_data(self, clinical_data_frame): 
       
        self.clinical_data = clinical_data_frame[clinical_data_frame['ID'] == self.patient_id]
    
    
def is_valid_tempcurve(temp_curve_df): 
    
    if temp_curve_df['Time'].iloc[-1] < 100:    # filter by length of temperaturecurve
        return False   
    
    if min(temp_curve_df['Temperature'])>0:     # filter curves that never actually cool down
        return False
    
    return True 