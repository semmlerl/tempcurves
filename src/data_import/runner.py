import pandas as pd
import os

import pickle as pickle
from class_patient_list import class_patient_list

def main(): 
    
    # Define the folder path containing the temperature data files
    folder_path = '../../../../data/tempcurves'
    
    # Generate a list of files starting with 'PKK' in the specified folder
    uploaded_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.xlsx')][0:10]
    print(f'Found {len(uploaded_files)} files')
  
    # generate the patient list by initializing an instance of the class patient for each file, 
        ## then initialize an instance of the class tempcurve per patient
        ## extract all features per tempcurve per patient
    patient_list = class_patient_list(uploaded_files)   
    
    ## adds the data from the recurrence_file to each patient
    patient_list.add_clinical_data('../../../../data/recurrence.xlsx')
    
    
    extracted_features_df = pd.DataFrame(extracted_features_list)
    
    # Load recurrence data from the specified path
    recurrence_file_path = '../../../data/recurrence.xlsx'
    recurrence_data = pd.read_excel(recurrence_file_path)
    
    # Verify the integrity of the recurrence_data DataFrame
    print("Recurrence DataFrame structure:")
    print(recurrence_data.info())
    print("Recurrence DataFrame head:")
    print(recurrence_data.head())
    print("Number of unique patients in recurrence data:", recurrence_data['ID'].nunique())
    print("Total number of recurrences:", recurrence_data['Recurrence'].sum())
    
    
    # Verify the structure of the extracted features DataFrame
    print("Extracted Features DataFrame structure:")
    print(extracted_features_df.info())
    print("Extracted Features DataFrame head:")
    print(extracted_features_df.head())
    print("Number of unique patients in extracted features:", extracted_features_df['ID'].nunique())
    print("Total number of temperature curves:", num_temp_curves)
    print("Mean number of temperature curves per patient:", num_temp_curves / extracted_features_df['ID'].nunique())
    
    # Group extracted features by patient ID and calculate mean for each patient
    extracted_features_grouped = extracted_features_df.drop('trace_id', axis = 1).groupby('ID').mean().reset_index()
    
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
    
    # save extracted features to files 
    with open("../../../data/extracted/raw_data_list.p", 'wb') as f: 
        pickle.dump(raw_data_list, f, protocol = pickle.HIGHEST_PROTOCOL)

    with open("../../../data/extracted/extracted_features_list.p", 'wb') as f: 
        pickle.dump(extracted_features_list, f, protocol = pickle.HIGHEST_PROTOCOL)
     
    with open("../../../data/extracted/recurrence_data_with_features.p", 'wb') as f: 
        pickle.dump(recurrence_data_with_features, f, protocol = pickle.HIGHEST_PROTOCOL )
        
if __name__ == "__main__":
    main()