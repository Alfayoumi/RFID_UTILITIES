import json
from RFIDSignalSimulator import *

sample_rate = 2e6  # Example sample rate in Hz
num_tags = 2  # Number of tags to simulate; change this as needed
preamble=np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1]) # Example preamble array
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

# Save the result in a JSON file
json_filename = f"unique_state_distribution_num_tags_{num_tags}.json"
with open(json_filename, 'w') as json_file:
    json.dump(unique_state_distribution, json_file, indent=4)

print(f"Unique state distribution has been saved to '{json_filename}'.")
