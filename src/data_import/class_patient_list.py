from class_patient import class_patient
import os 
import pandas as pd

class class_patient_list: 
    
    ## initializes a list of the class patient with a patient per file in uploaded files 
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
                
            print(patient_id, "exported, ", num_patients, "of", len(uploaded_files), "have been extracted")
            
            self.patient_list.append(patient)
            
     # the recurrence file is opened, for each patient from the list, if present, the respective data is added       
    def add_clinical_data(self, recurrence_file_path): 
        
        recurrence_data = pd.read_excel(recurrence_file_path)
        
        # Verify the integrity of the recurrence_data DataFrame
        print("Recurrence DataFrame structure:")
        print(recurrence_data.info())
        print("Recurrence DataFrame head:")
        print(recurrence_data.head())
        print("Number of unique patients in recurrence data:", recurrence_data['ID'].nunique())
        print("Total number of recurrences:", recurrence_data['Recurrence'].sum())    
        
        try:  
            for patient in self.patient_list: 
                patient.add_clinical_data(recurrence_data)
        except: 
            print("no clinical data for " + patient.patient_id)
            
    def generate_data_frame_features_per_tempcurve(self): 
        extracted_features = []
        print(extracted_features)
        
        for patient in self.patient_list:
            for tempcurve in patient.tempcurve_list: 
                              
                ## take the tempcurve features
                data_dict = tempcurve.features
                
                ## add the patients vein count 
                data_dict.update({
                    "vein_count'": patient.vein_count                    
                    })
                
                ## adds the clinical data, if no clinical data is available, 
                clinical_data_dict = patient.clinical_data.to_dict('records')[0]
                data_dict.update(clinical_data_dict)
                
                ## adds the dictionary to the list
                extracted_features.append(data_dict)                     
        
        ## generates a dataframe from the whole data
        extracted_features_df = pd.DataFrame(extracted_features)
               
        return extracted_features_df
    
    def generate_data_frame_raw_data_per_tempcurve(self): 
        """
        Returns a dataframe with one line per tempcurve including all clinical features and the raw data, starting
        from the dipping point till the end

        """
        
        extracted_features = []       
        
        for patient in self.patient_list:
            for tempcurve in patient.tempcurve_list: 
                                           
                ## add the patients vein count 
                data_dict ={
                    "vein_count'": patient.vein_count                    
                    }
                
                ## adds the clinical data
                clinical_data_dict = patient.clinical_data.to_dict('records')[0]
                data_dict.update(clinical_data_dict)
                
                ## adds the cutted trace 
                cutted_trace_dict = tempcurve.cutted_trace['Temperature'].to_dict()
                cutted_trace_dict = {f't{i}': value for i, (key, value) in enumerate(cutted_trace_dict.items())}
                data_dict.update(cutted_trace_dict)
                
                ## adds the dictionary to the list
                extracted_features.append(data_dict)                     
        
        ## generates a dataframe from the whole data
        extracted_features_df = pd.DataFrame(extracted_features)      
        
        
        return extracted_features_df
    
    def plot_dipping_point(self, outpath): 
        
        _i = 0
        
        for patient in self.patient_list:             
            _i += 1
            patient.plot_dipping_point(outpath)
            print(f"{str(patient.patient_id)} has been saved. {_i} of {len(self.patient_list)} have been extracted")
            
    def plot_cutted_trace(self, outpath): 
        """ Plot the cutted traces for all patients. 
        
        Arguments: 
            - outpath
        """        
        # counter for the loop
        _i = 0
        
        for patient in self.patient_list: 
            _i += 1
            patient.plot_cutted_trace(outpath)
            print(f"{str(patient.patient_id)} has been saved. {_i} of {len(self.patient_list)} have been extracted")