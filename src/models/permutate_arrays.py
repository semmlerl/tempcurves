import numpy as np
import itertools
import random


    num_patients, num_tempcurves, num_timepoints, num_channels = array.shape
    num_permutations = 100
    
    permutations = generate_random_permutations(num_tempcurves, num_permutations)
    
    # Initialize a list to collect the extended arrays
    extended_arrays = []
    
    # Iterate over each patient
    for patient_idx in range(num_patients):
        # Iterate over each permutation
        for perm in permutations:
            # Reorder the tempcurves dimension according to the current permutation
            reordered_array = array[patient_idx, perm, :, :]
            # Add the reordered array to the extended arrays list
            extended_arrays.append(reordered_array)
    
    # Convert the extended arrays list to a numpy array
    extended_array = np.array(extended_arrays)
    
    # Reshape the extended array to have the correct new dimensions
    extended_array = extended_array.reshape(-1, num_tempcurves, num_timepoints, num_channels)
    
# Example usage
# Original 4D numpy array with shape (patient_id, tempcurves, timepoint, channel)
original_array = np.random.rand(2, 3, 4, 5)  # Example array with random values

# Extend the array with a limited number of permutations
extended_array = extend_array_with_limited_permutations(original_array, num_permutations=10)

print("Original shape:", original_array.shape)
print("Extended shape:", extended_array.shape)

def generate_random_permutations(n, k):
    permutations = set()
    while len(permutations) < k:
        perm = tuple(random.sample(range(n), n))
        permutations.add(perm)
    return list(permutations)
