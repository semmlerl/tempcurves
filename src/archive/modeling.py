    # Prepare the dataset for training
    # Replace missing values with the median value for numeric columns only
    numeric_columns = recurrence_data_with_features.select_dtypes(include=[np.number]).columns
    recurrence_data_with_features[numeric_columns] = recurrence_data_with_features[numeric_columns].fillna(recurrence_data_with_features[numeric_columns].median())
    
    # Check for and handle infinite or excessively large values
    # Replace infinities with NaN and then fill NaNs with the median value
    recurrence_data_with_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Clip values that are excessively large
    max_float = np.finfo(np.float32).max
    recurrence_data_with_features[numeric_columns] = recurrence_data_with_features[numeric_columns].clip(upper=max_float)
    
    recurrence_data_with_features[numeric_columns] = recurrence_data_with_features[numeric_columns].fillna(recurrence_data_with_features[numeric_columns].median())
    
    # Add specified parameters to the model
    specified_parameters = ['Sex (m=0 / f=1)', 'Height', 'Weight', 'Age', 'Time_to_recurrence', 'LV_EF', 'LA_FLAECHE', 'KREATININ', 'initial_rhythm (1=SR, 2=AF)']
    feature_columns = list(recurrence_data_with_features.columns.difference(['Recurrence (1 = yes; 0 = no)', 'ID', 'Reconduction_Site', 'Normalized_Reconduction_Site']))
    feature_columns += specified_parameters
    
    # Ensure that feature columns are in X before training
    X = recurrence_data_with_features[feature_columns]
    y = recurrence_data_with_features['Normalized_Reconduction_Site']
    
    # Debugging steps: Print useful information
    print("Number of recurrences:", recurrence_data_with_features['Recurrence (1 = yes; 0 = no)'].sum())
    print("Number of male patients:", (recurrence_data_with_features['Sex (m=0 / f=1)'] == 0).sum())
    print("Number of female patients:", (recurrence_data_with_features['Sex (m=0 / f=1)'] == 1).sum())
    
    # Summary statistics for numeric features
    print("Summary statistics for numeric features:")
    print(recurrence_data_with_features[numeric_columns].describe().transpose())
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Log the results
    logging.info(f'Mean Squared Error: {mse}')
    logging.info(f'R2 Score: {r2}')
    
    # Additional debugging information
    print("Training set size:", X_train.shape[0])
    print("Testing set size:", X_test.shape[0])
    print("Training R2:", model.score(X_train, y_train))
    print("Testing R2:", r2)
    
    # Plot true vs. predicted values
    plt.figure()
    plt.scatter(y_test, y_pred, color='darkorange', label='Predicted vs. True')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs. Predicted Values')
    plt.legend(loc='upper left')
    plt.show()
    
    # Plot histogram of residuals
    residuals = y_test - y_pred
    plt.figure()
    plt.hist(residuals, bins=20, color='gray', edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Histogram of Residuals')
    plt.show()
    
    # Visualize a single tree from the forest
    plt.figure(figsize=(20, 10))
    plot_tree(model.estimators_[0], filled=True, feature_names=feature_columns, rounded=True, precision=1)
    plt.title("Visualization of a Single Tree from the RandomForestRegressor")
    plt.show()