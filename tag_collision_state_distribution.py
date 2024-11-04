import json
from RFIDSignalSimulator import *
import itertools
import random
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

# Define your TagConfig, SimulationConfig, and RFIDSignalSimulator classes here or import them as needed

def simulate_unique_states(sample_rate, preamble, snr_range, rn16_permutation):
    """
    Simulate the unique states for a given RN16 permutation and return the count of unique elements.

    Args:
        sample_rate (float): The sampling rate for the signal.
        preamble (np.array): The modulated preamble for the RFID signal.
        snr_range (tuple): Range for SNR values to use in the simulation.
        rn16_permutation (tuple): A tuple of RN16 values for the simulation.

    Returns:
        int: The count of unique elements from the simulation.
    """
    # Create tag configurations with the current RN16 permutation
    tag_configs = [
        TagConfig(
            distance=random.uniform(0.1, 3.0),
            snr=random.uniform(*snr_range),
            doppler_shift=random.uniform(-150, 150),
            multipath_fading=random.choice([True, False]),
            phase_rotation=random.uniform(0, 360),
            rn16=np.array(rn16)
        )
        for rn16 in rn16_permutation
    ]

    # Create and run simulator with the specified tag configurations
    sim_config = SimulationConfig(
        sample_rate=sample_rate,
        blf=40e3,
        snr=random.uniform(*snr_range),
        tag_configs=tag_configs,
        preamble=preamble
    )
    simulator = RFIDSignalSimulator(sim_config)

    # Simulate and count unique states
    num_unique_elements, _, _ = simulator.count_unique_states()
    return num_unique_elements

def get_unique_state_distribution(sample_rate, num_tags, preamble, snr_range):
    """
    Calculate the distribution of unique states across all possible RN16 permutations for a given number of tags.

    Args:
        sample_rate (float): The sampling rate for the signal.
        num_tags (int): The number of tags to simulate.
        preamble (np.array): The modulated preamble for the RFID signal.
        snr_range (tuple): Range for SNR values to use in the simulation.

    Returns:
        dict: A dictionary with `num_unique_elements` as keys and their occurrence counts as values.
    """
    unique_state_counts = defaultdict(int)  # Track frequency of each unique state count

    # Generate all possible RN16 permutations for the specified number of tags
    possible_rn16_values = list(itertools.product([0, 1], repeat=16))  # All 16-bit binary sequences
    rn16_permutations = itertools.permutations(possible_rn16_values, num_tags)

    # Total number of permutations (for verification)
    expected_permutations_count = sum(1 for _ in itertools.permutations(possible_rn16_values, num_tags))
    processed_permutations_count = 0
    print(expected_permutations_count)

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        future_to_permutation = {
            executor.submit(simulate_unique_states, sample_rate, preamble, snr_range, perm): perm
            for perm in rn16_permutations
        }

        for future in as_completed(future_to_permutation):
            num_unique_elements = future.result()
            unique_state_counts[num_unique_elements] += 1
            processed_permutations_count += 1  # Increment count

    # Verify all permutations were processed
    if processed_permutations_count == expected_permutations_count:
        print("All permutations processed successfully.")
    else:
        print(f"Warning: {expected_permutations_count - processed_permutations_count} permutations were missed.")

    return dict(unique_state_counts)


# Example Usage
sample_rate = 2e6  # Example sample rate in Hz
num_tags = 2  # Number of tags to simulate; change this as needed
preamble = np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1])  # Example preamble array
snr_range = (30, 60)  # Example SNR range

# Call the function
unique_state_distribution = get_unique_state_distribution(
    sample_rate=sample_rate,
    num_tags=num_tags,
    preamble=preamble,
    snr_range=snr_range
)

# Display the results
print("Unique State Distribution:")
for num_unique_elements, count in unique_state_distribution.items():
    print(f"{num_unique_elements} unique elements: {count} occurrences")

# Save the result in a JSON file with num_tags in the filename
json_filename = f"unique_state_distribution_num_tags_{num_tags}.json"
with open(json_filename, 'w') as json_file:
    json.dump(unique_state_distribution, json_file, indent=4)

print(f"Unique state distribution has been saved to '{json_filename}'.")

