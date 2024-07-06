import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, MaxPooling1D, Flatten
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def main(): 

    with open("../../../../data/extracted/extracted_raw_data_df.p", 'rb') as f: data = pickle.load(f)
     
    ##splitting into features and labels 
    features_df = data.iloc[:,13:]
    labels_df = data['Recurrence']
    
    # handling nan values
    features_df = features_df.fillna(40)   
    features_df.isna().sum()
    
    # normalizing features data using min_max_normalizer 
    features_df = (features_df + 60)/100
    
    ## replacing all values > 1 with 1
    features_df = features_df.applymap(lambda x: 1 if x > 1 else x)
    
    ## replacing all values <0 with 0
    features_df = features_df.applymap(lambda x: 0 if x < 0 else x)    
    
    # Splitting the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features_df, labels_df, test_size=0.2, random_state=None)
    
    # Convert DataFrames to numpy arrays for TensorFlow
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
  
    # Define the neural network model
    model = Sequential([
        Input(shape =(X_train.shape[1],1)),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(), 
        Dense(256, activation='relu'),
        Dropout(0.5), 
        Dense(256, activation='relu'),
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
    
    plot_confusion_matrix(y_pred_prob, y_test, cutoff = 0.3)
    
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