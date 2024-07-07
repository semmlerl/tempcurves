import numpy as np
import itertools
import random

def generate__permutated_array(x_train, y_train, num_permutations):
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
    
    return extended_array, y_labels
    

def generate_random_permutations(n, k):
    permutations = set()
    while len(permutations) < k:
        perm = tuple(random.sample(range(n), n))
        permutations.add(perm)
    return list(permutations)
