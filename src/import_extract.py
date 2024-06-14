import pandas as pd
import os
import json as json
import pickle as pickle
import import_extract_helpers as import_functions
import matplotlib.pyplot as plt

def main(): 
    
    # Define the folder path containing the temperature data files
    folder_path = '../../../data/tempcurves'
    
    # Generate a list of files starting with 'PKK' in the specified folder
    uploaded_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.xlsx')]
    print(f'Found {len(uploaded_files)} files')
    
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
    
    # Initialize a DataFrame to store extracted features for each patient
    extracted_features_list = []
    raw_data_list=[]
    
    
    # Process each temperature curve file
    num_temp_curves = 0
    num_patients = 0
    for file in uploaded_files:
        
        patient_id = os.path.splitext(os.path.basename(file))[0]
        xls = pd.ExcelFile(file)
        
        patient_vein_count = 0
        
        for sheet_name in xls.sheet_names:
            
            temp_curve_df = pd.read_excel(xls, sheet_name=sheet_name, usecols=[0, 1])  # Only load the first two columns
            temp_curve_df.columns = ['Time', 'Temperature']  # Ensure columns are named correctly
           
                        
            if import_functions.is_valid_tempcurve(temp_curve_df):    
                
                ##cutting the temperature curve at the dipping point to exclude the nonspecific "waiting time at the beginning"
                temp_curve_df = import_functions.find_dipping_point(temp_curve_df)
                
                ##plotting cutted files
                plt.plot(temp_curve_df['Temperature'])
                plt.savefig("../../../data/dipping/" + patient_id + str(patient_vein_count) + ".png")
                plt.close()
                             
                patient_vein_count += 1 # add a counter to create unique combination of patient and vein 
                
                ## extract features and add them to feature_list
                features = import_functions.extract_features(temp_curve_df)
                features['ID'] = patient_id
                features['trace_id'] = patient_id + str(patient_vein_count)
                extracted_features_list.append(features)
                
                ## extract raw data and add it to raw_data_list
                raw_data = {}
                raw_data['id'] = patient_id 
                raw_data['trace_id'] = patient_id + str(patient_vein_count)
                raw_data['data'] = temp_curve_df
                raw_data_list.append(raw_data)
                
                num_temp_curves += 1
            
        num_patients +=1
            
        print(num_patients, "of", len(uploaded_files), "have been extracted")
    
    extracted_features_df = pd.DataFrame(extracted_features_list)
    
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