from class_patient import class_patient
import os 
import pandas as pd

class class_patient_list: 
    
    def __init__(self, uploaded_files): 
        
        self.patient_list = []
        
        ## generate a counter for the total number of patients 
        num_patients = 0
        
        for file in uploaded_files:
            
            patient_id = os.path.splitext(os.path.basename(file))[0]
            xls = pd.ExcelFile(file)
            
            # generate a new instance of the class patient and add all tempcurves
            patient = class_patient(patient_id, xls)      

            # calculates the dipping point for each tempcurve of the patient
            patient.calculate_dipping_points()
            
            # extracts the features of each tempcurve of the patient
            patient.extract_features()  
            
            num_patients +=1
                
            print(num_patients, "of", len(uploaded_files), "have been extracted")
            
            self.patient_list.append(patient)
            
            
    def add_clinical_data(self, recurrence_file_path): 
        
        recurrence_data = pd.read_excel(recurrence_file_path)
        
        # Verify the integrity of the recurrence_data DataFrame
        print("Recurrence DataFrame structure:")
        print(recurrence_data.info())
        print("Recurrence DataFrame head:")
        print(recurrence_data.head())
        print("Number of unique patients in recurrence data:", recurrence_data['ID'].nunique())
        print("Total number of recurrences:", recurrence_data['Recurrence'].sum())    
        
        for patient in self.patient_list: 
            patient.add_clinical_data(recurrence_data)
        
        
        