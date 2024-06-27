import os 

import pickle as pickle
from class_patient_list import class_patient_list

def main(): 
    
    # Define the folder path containing the temperature data files
    folder_path = '../../../../data/tempcurves'
    
    # Generate a list of files starting with 'PKK' in the specified folder
    uploaded_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.xlsx')]
    print(f'Found {len(uploaded_files)} files')
  
    # generate the patient list by initializing an instance of the class patient for each file, 
        ## then initialize an instance of the class tempcurve per patient
        ## extract all features per tempcurve per patient
    patient_list = class_patient_list(uploaded_files)   
    
    ## adds the data from the recurrence_file to each patient
    patient_list.add_clinical_data('../../../../data/recurrence.xlsx')  
    
    ## plots the raw data and the dipping point
    #patient_list.plot_dipping_point("../../../../data/plot_raw/")
    
    ## plots the cutted trace and a selection of features
    patient_list.plot_cutted_trace("../../../../data/plot_cutted/")
     
    # generates a dataframe with one line per tempcurve including:
        # tempcurve features
        # patients vein count
        # patients clinical data
    extracted_features_df = patient_list.generate_data_frame_features_per_tempcurve()
    
     # Verify the structure of the extracted features DataFrame
    print("Extracted Features DataFrame structure:")
    print(extracted_features_df.info())
    print("Extracted Features DataFrame head:")
    print(extracted_features_df.head())
  
    # Group extracted features by patient ID and calculate mean for each patient
    extracted_features_grouped = extracted_features_df.groupby('ID').mean().reset_index()
    
    with open("../../../../data/extracted/extracted_features_df.p", 'wb') as f: pickle.dump(extracted_features_df, f, protocol = pickle.HIGHEST_PROTOCOL)
     
    with open("../../../../data/extracted/extracted_features_grouped.p", 'wb') as f:  pickle.dump(extracted_features_grouped, f, protocol = pickle.HIGHEST_PROTOCOL )
        
if __name__ == "__main__":
    main()