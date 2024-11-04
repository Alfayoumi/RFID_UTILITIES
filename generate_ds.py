from RFIDSignalSimulator import *


generate_collision_dataset(
    num_samples_per_class=20000,
    num_classes=[0, 1, 2, 3, 4],
    sample_rate=2e6,
    noise_level=0.1,
    preamble=np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1]),
    filename="custom_collision_database.csv",
    snr_range=(30, 60),
    ambiguous_snr_range=(0, 10),
    high_tag_count_range=(20, 100)
)