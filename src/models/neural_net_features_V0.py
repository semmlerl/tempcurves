import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Normalization

def main(): 

    with open("../../../../data/extracted/extracted_features_df.p", 'rb') as f: data = pickle.load(f)
    
    features_df = data.iloc[:,0:18]
    
    for column in features_df.columns:
        features_df[column] = (features_df[column] - min(features_df[column]))/(max(features_df[column]) - min(features_df[column]))
          
    labels_df = data['Recurrence']
    # Splitting the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features_df, labels_df, test_size=0.2, random_state=42)
    
    # Convert DataFrames to numpy arrays for TensorFlow
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

  
    # Define the neural network model
    model = Sequential([        
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')  # Assuming binary classification
    ])    
   
    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    
    # Evaluate the model on test data
    test_loss, test_acc = model.evaluate(X_test, y_test)
    
    y_pred_prob = model.predict(X_test)
    
    print(f"Test accuracy: {test_acc}")
    
if __name__ == "__main__":
    main()