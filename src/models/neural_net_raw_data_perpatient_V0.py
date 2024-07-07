import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, MaxPooling2D, Flatten
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def main(): 

    with open("../../../../data/extracted/extracted_raw_data_df.p", 'rb') as f: data = pickle.load(f)    
    
    # generating the array holding the raw data per patient
    array = format_raw_data_array(data)
             
    # generating the dataframe holding the recurrence 
    labels_df = data.groupby('ID')['Recurrence'].mean().reset_index()['Recurrence']
      
    # normalizing features data using min_max_normalizer 
    array = (array + 60)/100
    
    ## replacing all values > 1 with 1
    array[array > 1] = 1    
    
    ## replacing all values <0 with 0
    array[array < 0] = 0       
    
    # Splitting the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(array, labels_df, test_size=0.2, random_state=None)
    
    # Convert DataFrames to numpy arrays for TensorFlow
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
  
    # Define the neural network model
    model = Sequential([
        Input(shape =(X_train.shape[1],X_train.shape[2],1)),
        Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
        MaxPooling2D(pool_size=(8,8)),
        Flatten(), 
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])    
    
    model.summary()
   
    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2)
    
    # Evaluate the model on test data
    test_loss, test_acc = model.evaluate(X_test, y_test)
    
    y_pred_prob = model.predict(X_test).flatten()
    
    print(f"Test accuracy: {test_acc}")
    
    plot_confusion_matrix(y_pred_prob, y_test, cutoff = 0.2)
    
    plot = pd.DataFrame({'pred': y_pred_prob, 'true': y_test})
    
    plot.groupby('true')['pred'].hist(alpha = 0.5, legend = True)
    
if __name__ == "__main__":
    main()
    
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
        
    return array