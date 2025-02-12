import pandas as pd
from class_tempcurve import class_tempcurve
import math
import matplotlib.pyplot as plt


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

            if len(temp_curve_df.columns) < 2: 
                continue 

            # renames files properly 
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
        
    def plot_dipping_point(self, outpath): 
        
        for tempcurve in self.tempcurve_list: 
            tempcurve.plot_dipping_point(outpath)  
    
    def plot_cutted_trace(self, outpath):         
        """
        Generates a plot with subplots for all tempcurves of a patient. 
        Fhe cutted trace and features are displayed
        
        Arguments: 
            - outpath: the path to the folder that the generated image will be saved to
        """
        
        # calculating the amount of rows         
        ncol = 4
        nrow = math.ceil(len(self.tempcurve_list)/ncol)
        
        ## checking that there is any data to plot and exiting the function if there is none
        if nrow == 0: 
            print(str(self.patient_id) + " has no data to be plotted")
            return None 
        
        ## generating the figure
        fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize = (20, 4 * nrow ))       
        
        # Assign each axes to the respective plot 
        for ax, tempcurve in zip(axes.flatten(), self.tempcurve_list):        
            tempcurve.plot_cutted_trace(ax)        

        # Adjust layout
        fig.suptitle((str(self.patient_id) + ": " + "Recurrence: " + str(self.clinical_data['Recurrence'].iloc[0])), fontsize = 16)
        
        plt.tight_layout()        
        
        plt.savefig(outpath + str(self.patient_id) +  ".png")  
        
        ## close the figure
        plt.close()            
            
def is_valid_tempcurve(temp_curve_df): 
    
    if temp_curve_df['Time'].iloc[-1] < 100:    # filter by length of temperaturecurve
        return False   
    
    if min(temp_curve_df['Temperature'])>0:     # filter curves that never actually cool down
        return False
    
    return True 

