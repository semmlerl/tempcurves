import numpy as np
import keras
import pickle
from sklearn.model_selection import train_test_split
from keras import layers
from keras.layers import Conv1D, MaxPooling1D
from matplotlib import pyplot as plt
from MILAttentionLayer import MILAttentionLayer
from metrics import calc_plot_roc, plot_predict, plot_confusion_matrix

plt.style.use("ggplot")

def format_raw_data_(path): 
    """
    Takes the input dataframe with a line per tempcurve, outputs a numpy 3d array: 
        - dimensions: number_of_patients, maximum number of tempcurves per patient, maximum length of tempcurve in seconds, channel (tensorflow cnn need a 3 channel)
        - all missing data (e.g. shorter tempcurves or lower number of tempcurves is set to 40°C as the maximum temperature )
        
    replaces all NaN by 40, normalizing data using min max, replacing all lower and higher values 

    """
      
    ## loading the raw data 
    with open(path, 'rb') as f: data = pickle.load(f)  
    
    
    # filter curves with more than 13 or less than 4 temperature curves
    data = data[data["vein_count'"]< 13]
    data = data[data["vein_count'"]> 3]    
    
    ## formatting raw data into 3d array per patient 
    raw_data = data.iloc[:,4 :664].join(data['ID'])
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
     
    # generating the dataframe holding the recurrence 
    labels_df = np.array(data.groupby('ID')['Recurrence'].mean().reset_index()['Recurrence'])

    # Splitting the data into training and test sets
    x_train, x_val, y_train, y_val = train_test_split(array, labels_df, test_size=0.2, random_state=42)

    x_train = list(np.swapaxes(x_train, 0, 1))
    x_val = list(np.swapaxes(x_val, 0, 1))    
         
    return x_train, x_val, y_train, y_val
        
def create_model(instance_shape, bag_size):
    # Extract features from inputs.
    inputs, embeddings = [], []
    shared_dense_layer_1 = layers.Dense(128, activation="relu")
    shared_dense_layer_2 = layers.Dense(64, activation="relu")
    for _ in range(bag_size):
        inp = layers.Input(instance_shape)
        flatten = layers.Flatten()(inp)
        dense_1 = shared_dense_layer_1(flatten)
        dense_2 = shared_dense_layer_2(dense_1)
        inputs.append(inp)
        embeddings.append(dense_2)

    # Invoke the attention layer.
    alpha = MILAttentionLayer(
        weight_params_dim=256,
        kernel_regularizer=keras.regularizers.L2(0.01),
        use_gated=True,
        name="alpha",
    )(embeddings)

    # Multiply attention weights with the input layers.
    multiply_layers = [
        layers.multiply([alpha[i], embeddings[i]]) for i in range(len(alpha))
    ]

    # Concatenate layers.
    concat = layers.concatenate(multiply_layers, axis=1)
    
    # added dense layer
    dense_3 = layers.Dense(128, activation = "relu")(concat)

    # Classification output node.
    output = layers.Dense(1, activation="sigmoid")(dense_3)

    return keras.Model(inputs, output)

def create_model_CNN(instance_shape, bag_size):
    # Extract features from inputs.
    inputs, embeddings = [], []
    
    conv1_1 = Conv1D(16, kernel_size=(3), activation='relu') 
    mpool_1 = MaxPooling1D(2)       
    
    shared_dense_layer_1 = layers.Dense(64, activation="relu")
    
    for _ in range(bag_size):
        inp = layers.Input(instance_shape)
        x = conv1_1(inp)
        x = mpool_1(x)
        flatten = layers.Flatten()(x)
        dense_1 = shared_dense_layer_1(flatten)

        inputs.append(inp)
        embeddings.append(dense_1)

    # Invoke the attention layer.
    alpha = MILAttentionLayer(
        weight_params_dim=256,
        kernel_regularizer=keras.regularizers.L2(0.01),
        use_gated=True,
        name="alpha",
    )(embeddings)

    # Multiply attention weights with the input layers.
    multiply_layers = [
        layers.multiply([alpha[i], embeddings[i]]) for i in range(len(alpha))
    ]

    # Concatenate layers.
    concat = layers.concatenate(multiply_layers, axis=1)
    
    # added dense layer
    dense_3 = layers.Dense(128, activation = "relu")(concat)

    # Classification output node.
    output = layers.Dense(1, activation="sigmoid")(dense_3)

    return keras.Model(inputs, output)

def train(train_data, train_labels, val_data, val_labels, model):
    # Train model.
    # Prepare callbacks.
    # Path where to save best weights.

    # Take the file name from the wrapper.
    file_path = "/tmp/best_model.weights.h5"

    # Initialize model checkpoint callback.
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        file_path,
        monitor="val_loss",
        verbose=2,
        mode="min",
        save_best_only=True,
        save_weights_only=True,
    )

    # Initialize early stopping callback.
    # The model performance is monitored across the validation data and stops training
    # when the generalization error cease to decrease.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, mode="min"
    )

    # Compile model.
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # Fit model.
    model.fit(
        train_data,
        train_labels,
        validation_data=(val_data, val_labels),
        epochs=10,   
        batch_size=12,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1,
    )

    # Load best weights.
    model.load_weights(file_path)

    return model

def predict(data, labels, trained_model):
    # Collect info per model.
    models_predictions = []
    models_attention_weights = []
    models_losses = []
    models_accuracies = []

    # Predict output classes on data.
    predictions = trained_model.predict(data)
    models_predictions.append(predictions)

    # Create intermediate model to get MIL attention layer weights.
    intermediate_model = keras.Model(trained_model.input, trained_model.get_layer("alpha").output)

    # Predict MIL attention layer weights.
    intermediate_predictions = intermediate_model.predict(data)

    attention_weights = np.squeeze(np.swapaxes(intermediate_predictions, 1, 0))
    models_attention_weights.append(attention_weights)

    loss, accuracy = trained_model.evaluate(data, labels, verbose=0)
    models_losses.append(loss)
    models_accuracies.append(accuracy)

    print(
        f"The average loss and accuracy are {np.sum(models_losses, axis=0):.2f}"
        f" and {100 * np.sum(models_accuracies, axis=0):.2f} % resp."
    )

    return (
        np.sum(models_predictions, axis=0),
        np.sum(models_attention_weights, axis=0),
    )

# %%
 
    BAG_SIZE = 12    
    
    ## loading the raw data     
    x_train, x_val, y_train, y_val = format_raw_data_("../../../data/extracted/extracted_raw_data_df.p")        
      
    # Building model(s).
    instance_shape = x_train[0][0].shape
    model = create_model_CNN(instance_shape, BAG_SIZE)
    
    # Show single model architecture.
    print(model.summary())
    
    # Training model(s).
    trained_model = train(x_train, y_train, x_val, y_val, model)
        
    # Evaluate and predict classes and attention scores on validation data.
    class_predictions, attention_params = predict(x_val, y_val, trained_model)

# %%
    # calculating roc 
    calc_plot_roc(y_val, class_predictions)
    
    # plotting prediction table
    plot_predict(y_val, class_predictions)
    
    # plotting confusion matrix
    plot_confusion_matrix(y_val, class_predictions, cutoff = 0.47)

#%%
if __name__ == "__main__": 
    main()
# %%
