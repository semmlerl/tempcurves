import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def main(): 

    with open("../../../../data/extracted/extracted_features_df.p", 'rb') as f: data = pickle.load(f)
    
     
    ##splitting into features and labels 
    features_df = data.iloc[:,0:18].join(data['Reconduction_Site'])
    
    # normalizing features data using min_max_normalizer 
    for column in features_df.columns:
        features_df[column] = (features_df[column] - min(features_df[column]))/(max(features_df[column]) - min(features_df[column]))
    
     # handling nan values
    features_df = features_df.fillna(1)   
    features_df.isna().sum()
    
    labels_df = data['Recurrence']
    # Splitting the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features_df, labels_df, test_size=0.2, random_state=None)
    
    # Convert DataFrames to numpy arrays for TensorFlow
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
  
    # Define the neural network model
    model = Sequential([
        Input(shape =(X_train.shape[1],)),       
        Dense(64, activation='relu'),
        Dropout(0.5),
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
    
    plot_confusion_matrix(y_pred_prob, y_test)
    
    plot = pd.DataFrame({'pred': y_pred_prob, 'true': y_test})
    
    plot.groupby('true')['pred'].hist(alpha = 0.5, legend = True)
    
if __name__ == "__main__":
    main()
    
def plot_confusion_matrix(predicted, actual): 
    predicted = tf.squeeze(predicted)
    predicted = np.array([1 if x >= 0.3 else 0 for x in predicted])
    actual = np.array(actual)
    conf_mat = confusion_matrix(actual, predicted)
    displ = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    displ.plot()