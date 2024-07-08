import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import random
import math

def main(): 

    with open("../../../../data/extracted/extracted_raw_data_df.p", 'rb') as f: data = pickle.load(f)    
    
    # generating the array holding the raw data per patient
    array = format_raw_data_array(data)
                 
    # generating the dataframe holding the recurrence 
    labels_df = data.groupby('ID')['Recurrence'].mean().reset_index()['Recurrence']    
    
    # Splitting the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(array, labels_df, test_size=0.2, random_state=42)
    
    # expanding the training data_set via permutation
    X_train, y_train = generate__permutated_array(X_train, y_train, 100)
    
    # Convert DataFrames to numpy arrays for TensorFlow
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
  
    # Define the neural network model
    model = Sequential([
        Input(shape =(X_train.shape[1],X_train.shape[2],1)),
        Conv2D(filters=32, kernel_size=(1,3), activation='relu'),
        MaxPooling2D(pool_size=(8,8)),
        Flatten(), 
        Dense(256, activation='relu'),
        Dropout(.5), 
        Dense(64), 
        Dense(1, activation='sigmoid')
    ])    
    
    # output a model summary
    model.summary()
   
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    
    # Evaluate the model on test data
    test_loss, test_acc = model.evaluate(X_test, y_test)
    y_pred_prob = model.predict(X_test).flatten()
   
    # calculate the roc metric
    calculate_roc(y_test, y_pred_prob)  
  
    # plot distribution of predictions
    plot = pd.DataFrame({'pred': y_pred_prob, 'true': y_test})
    plot.groupby('true')['pred'].hist(alpha = 0.5, legend = True)
    
    # plot confusion matrix
    plot_confusion_matrix(y_pred_prob, y_test, cutoff = 0.2)
      
def plot_confusion_matrix(predicted, actual, cutoff = 0.5): 
    """
    Generates a confusion matrix for the respective model 

    Parameters
    ----------
    predicted : dataframe
        vector carrying the model predictions
    actual : dataframe
        dataframe carrying the actual label
    cutoff : number
        value above which prediction  = 1
    """
    
    predicted = tf.squeeze(predicted)
    predicted = np.array([1 if x >= cutoff else 0 for x in predicted])
    actual = np.array(actual)
    conf_mat = confusion_matrix(actual, predicted)
    displ = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    displ.plot()
      
def format_raw_data_array(data): 
    """
    Takes the input dataframe with a line per tempcurve, outputs a numpy 3d array: 
        - dimensions: number_of_patients, maximum number of tempcurves per patient, maximum length of tempcurve in seconds, channel (tensorflow cnn need a 3 channel)
        - all missing data (e.g. shorter tempcurves or lower number of tempcurves is set to 40°C as the maximum temperature )
        
    replaces all NaN by 40, normalizing data using min max, replacing all lower and higher values 

    """
       
    ## formatting raw data into 3d array per patient 
    raw_data = data.iloc[:,13 :441].join(data['ID'])
    raw_data.fillna(40, inplace = True)
    
    unique_ids = raw_data['ID'].unique()
    num_ids = len(unique_ids)
    
    # Number of columns (excluding the ID column)
    num_columns = raw_data.shape[1] - 1
    
    # Get maximum number of rows for any id
    max_rows = raw_data.groupby('ID').size().max()
    
    # Initialize the 3D numpy array with the fill value 40
    array = np.full((num_ids, max_rows, num_columns, 1), fill_value=40)
    
    # Fill the array with the data from the dataframe
    for i, uid in enumerate(unique_ids):
        # Select rows for the current id
        rows = raw_data[raw_data['ID'] == uid].iloc[:, :-1].values
    
        # Fill the corresponding slice in the 3D array
        array[i, :rows.shape[0], :, 0] = rows  
    
    # normalizing features data using min_max_normalizer 
    array = (array + 60)/100
    
    ## replacing all values > 1 with 1
    array[array > 1] = 1    
    
    ## replacing all values <0 with 0
    array[array < 0] = 0     
        
    return array

def calculate_roc(true_labels, predicted_labels): 
    """
    creates a ROC curve for a binary classifier

    input: 
        - predicted_labels
        - true_labels

    """
    # Generate ROC curve
    fpr, tpr, _ = roc_curve(true_labels, predicted_labels)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    (plt.figure(),
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc),
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--'),
    plt.xlim([0.0, 1.0]),
    plt.ylim([0.0, 1.05]),
    plt.xlabel('False Positive Rate'),
    plt.ylabel('True Positive Rate'),
    plt.title('Receiver Operating Characteristic'),
    plt.legend(loc="lower right"))
    plt.show()

    print(f"AUC: {roc_auc}")

def generate__permutated_array(X_train, y_train, num_permutations):
    """
    Given the array X_train, the function generates num_permutations permutations of the tempcurves of each patient and builds a new 
    array with these, which is num_permutations * the number of patients. It does a few things to be noted: 
        -it only permutates the actual tempcurves but not the placeholder rows for patients who have less than the maxmimum number of tempcurves. 
        aka each permutation starts with tempcurves in some order and ends with zero lines. 
        - if the number of tempcurves per patients does not allow sufficient permutations (4 tempcurves only allow 4! = 24 tempcurves), 
        there will be less permutations to exclude copies 
    
    Parameters
    ----------
    X_train : the array of training data
    y_train : the series of lables
    num_permutations : the number of permutations to create per patient 

    Returns
    -------
    extended_array : the extended array of training data       
    y_labels : the extended series of labels         
    """
    num_patients, num_tempcurves, num_timepoints, num_channels = X_train.shape        
    
    # Initialize a list to collect the extended arrays
    extended_arrays = []
    extended_labels = []
    
    # Iterate over each patient
    for patient_idx in range(num_patients):
        
        # get the number of tempcurves for that patient, so that the permutation is only done on actual tempcurves
        row_means = np.mean(X_train[patient_idx,:,:,:], axis=1)
        
        try: 
            first_row_index = np.where(row_means == 1)[0][0]
        except: 
            first_row_index = num_tempcurves
        
        ## generate permutations of the tempcurves 
        permutations = generate_random_permutations(first_row_index, num_permutations)
        
        # Iterate over each permutation
        for perm in permutations:
            
            # Initialize the 3D numpy array with the fill value 1
            reordered_array = np.full((num_tempcurves, num_timepoints, 1), dtype = float, fill_value=1)
            
            # Reorder the tempcurves dimension according to the current permutation            
            reordered_array[0:first_row_index, :, 0] = X_train[patient_idx, perm, :, 0]
            
            # Add the reordered array to the extended arrays list
            extended_arrays.append(reordered_array)
            
            # Add the label to the extended_labels
            extended_labels.append(y_train.iloc[patient_idx])
    
    # Convert the extended arrays list to a numpy array
    extended_array = np.array(extended_arrays)
    
    # Reshape the extended array to have the correct new dimensions
    extended_array = extended_array.reshape(-1, num_tempcurves, num_timepoints, num_channels)  
    
    # reformat extended_labels to Pandas series   
    extended_labels = pd.Series(extended_labels)

    # shuffle array of permutated training data and labels 
    shuffle = random.sample(range(len(extended_labels)),len(extended_labels) )
    
    extended_labels = extended_labels[shuffle]
    extended_array = extended_array[shuffle, :,:,:]

    return extended_array, extended_labels
    
def generate_random_permutations(n, k):
    
    """
    Given a range(n) this function returns k unique permutations of that range. If there is not enough permtuations, a shorter list will be returned

    Parameters
    ----------
    n : range 
    k : amount of permutations        

  
    """
    permutations = set()
    
    while len(permutations) < k and len(permutations) < math.factorial(n) :
        perm = tuple(random.sample(range(n), n))
        permutations.add(perm)
        
    return list(permutations)

def generate__permutated_array_v0(x_train, y_train, num_permutations):
    num_patients, num_tempcurves, num_timepoints, num_channels = x_train.shape
        
    permutations = generate_random_permutations(num_tempcurves, num_permutations)
    
    # Initialize a list to collect the extended arrays
    extended_arrays = []
    
    # Iterate over each patient
    for patient_idx in range(num_patients):
        # Iterate over each permutation
        for perm in permutations:
            # Reorder the tempcurves dimension according to the current permutation
            reordered_array = x_train[patient_idx, perm, :, :]
            # Add the reordered array to the extended arrays list
            extended_arrays.append(reordered_array)
    
    # Convert the extended arrays list to a numpy array
    extended_array = np.array(extended_arrays)
    
    # Reshape the extended array to have the correct new dimensions
    extended_array = extended_array.reshape(-1, num_tempcurves, num_timepoints, num_channels)
    
    # expand the labels by the respective length
    y_labels = y_train.repeat(num_permutations).reset_index(drop=True)
    
    # shuffle the extended dataset
    
    
    return extended_array, y_labels

if __name__ == "__main__":
    main()