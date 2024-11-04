from matplotlib import pyplot as plt
import random
import pandas as pd
import itertools
from collections import defaultdict
import numpy as np


class ModulationStrategy:
    def modulate(self, bit_sequence):
        raise NotImplementedError("Modulate method must be implemented.")


class FM0Modulation(ModulationStrategy):
    def modulate(self, bit_sequence):
        modulated_signal = []
        previous_state_high = True
        for bit in bit_sequence:
            if previous_state_high:
                if bit == 0:
                    modulated_signal.extend([1, 0])
                    previous_state_high = True
                elif bit == 1:
                    modulated_signal.extend([1, 1])
                    previous_state_high = False
            else:
                if bit == 0:
                    modulated_signal.extend([0, 1])
                    previous_state_high = False
                elif bit == 1:
                    modulated_signal.extend([0, 0])
                    previous_state_high = True
        return np.array(modulated_signal)


# Enhanced Strategy Pattern for Channel Effects
class EnhancedChannelEffect:
    def __init__(self, distance, snr, doppler_shift, multipath_fading, sample_rate, blf, num_tags,
                 attenuation_factor=1.0, max_reflections=3, reflection_attenuation_range=(0.02, 0.3),
                 multipath_delay_range=(0.01, 0.2), phase_shift_factor=2 * np.pi, phase_rotation=0):
        self.distance = distance
        self.snr = snr
        self.doppler_shift = doppler_shift
        self.multipath_fading = multipath_fading
        self.sample_rate = sample_rate
        self.blf = blf
        self.num_tags = num_tags  # Number of tags in the environment
        self.phase_rotation = phase_rotation  # Phase rotation (specific for each tag)

        # Configuration parameters for effects
        self.attenuation_factor = attenuation_factor / (distance ** 2)  # Attenuation based on distance
        self.max_reflections = max_reflections
        self.reflection_attenuation_range = reflection_attenuation_range
        self.multipath_delay_range = multipath_delay_range
        self.phase_shift_factor = phase_shift_factor

        # Calculate noise level from SNR
        self.noise_level = self.calculate_noise_level(snr) / self.num_tags

    @staticmethod
    def calculate_noise_level(snr):
        return 10 ** (-snr / 20)  # Convert SNR (in dB) to noise level

    @staticmethod
    def generate_noise(length, noise_level, dc_offset_i=1.0, dc_offset_q=1.0, gain_imbalance_q=1.05,
                       phase_imbalance_deg=0.0, multipath_fading=False):
        # Introduce a slight gain imbalance between I and Q components
        gain_imbalance_i = 1.0  # No gain change for I

        # Apply DC offsets and gain imbalance
        noise_i = (np.random.normal(0, noise_level, length) + dc_offset_i) * gain_imbalance_i
        noise_q = (np.random.normal(0, noise_level, length) + dc_offset_q) * gain_imbalance_q

        # Combine I and Q into a complex signal
        noise_signal = noise_i + 1j * noise_q

        # Apply phase imbalance by rotating the Q component slightly
        phase_imbalance_rad = np.deg2rad(phase_imbalance_deg)
        noise_signal = noise_signal * np.exp(1j * phase_imbalance_rad)

        if multipath_fading:
            noise_signal = EnhancedChannelEffect.apply_multipath_fading_static(noise_signal)

        return noise_signal

    def apply(self, signal):
        # Apply each effect in sequence
        signal = self.apply_attenuation(signal)
        signal = self.apply_phase_rotation(signal)  # Specific phase rotation for each tag
        return signal

    def apply_attenuation(self, signal):
        return signal * self.attenuation_factor

    def apply_doppler_shift(self, signal):
        if self.doppler_shift != 0:
            time = np.arange(len(signal)) / self.sample_rate
            doppler_effect = np.exp(1j * 2 * np.pi * self.doppler_shift * time)
            return signal * doppler_effect
        return signal

    def apply_phase_shift(self, signal):
        phase_shift = self.phase_shift_factor * self.distance / (3e8 / self.blf)  # Distance-based phase shift
        return signal * np.exp(1j * phase_shift)

    def apply_multipath_fading(self, signal):
        if self.multipath_fading:
            return self.apply_multipath_fading_static(signal)
        return signal

    def apply_noise(self, signal):
        noise = np.random.normal(0, self.noise_level, len(signal)) + 1j * np.random.normal(0, self.noise_level,
                                                                                           len(signal))
        return signal + noise

    def apply_phase_rotation(self, signal):
        if self.phase_rotation != 0:
            phase_rotation_matrix = np.exp(1j * np.deg2rad(self.phase_rotation))
            return signal * phase_rotation_matrix
        return signal

    @staticmethod
    def apply_multipath_fading_static(signal, max_reflections=3, reflection_attenuation_range=(0.02, 0.3),
                                      multipath_delay_range=(0.01, 0.2)):
        faded_signal = signal.copy()
        num_paths = np.random.randint(1, max_reflections + 1)  # Random number of reflection paths
        for _ in range(num_paths):
            delay_samples = int(np.random.uniform(multipath_delay_range[0], multipath_delay_range[1]) * len(signal))
            reflection_attenuation = np.random.uniform(*reflection_attenuation_range)  # Random attenuation
            multipath_signal = np.roll(signal, delay_samples) * reflection_attenuation
            faded_signal += multipath_signal
        return faded_signal


# Factory Pattern for Creating Tags and Readers
class RFIDFactory:
    @staticmethod
    def create_modulation(strategy_type):
        if strategy_type == 'FM0':
            return FM0Modulation()
        # Can add other modulation strategies here.

    @staticmethod
    def create_channel_effect(strategy_type, **kwargs):
        if strategy_type == 'Enhanced':
            return EnhancedChannelEffect(**kwargs)
        # Can add other channel effect strategies here.


# Tag Configuration Class
class TagConfig:
    def __init__(self, distance, snr, doppler_shift, multipath_fading, phase_rotation, rn16=None):
        self.distance = distance
        self.snr = snr
        self.doppler_shift = doppler_shift
        self.multipath_fading = multipath_fading
        self.phase_rotation = phase_rotation
        self.rn16 = rn16  # Predefined RN16 or None for random RN16


# Simulation Configuration Class
class SimulationConfig:
    def __init__(self, sample_rate, blf, snr, tag_configs, preamble):
        self.sample_rate = sample_rate
        self.blf = blf
        self.snr = snr
        self.tag_configs = tag_configs
        self.preamble = preamble


# Simplified Tag without State Management
class Tag:
    def __init__(self, tag_id, modulation_strategy, channel_effect_strategy, n_samples_TAG_BIT, rn16=None):
        self.tag_id = tag_id
        self.modulation_strategy = modulation_strategy
        self.channel_effect_strategy = channel_effect_strategy
        self.n_samples_TAG_BIT = n_samples_TAG_BIT
        self.rn16 = rn16 if rn16 is not None else self.generate_random_rn16()

    def generate_random_rn16(self):
        return np.random.randint(0, 2, 16)

    def respond_to_query(self, preamble, apply_channel_effect=True):
        # Modulate the RN16
        modulated_rn16 = self.modulation_strategy.modulate(self.rn16)

        # Concatenate the preamble (already modulated) and RN16
        full_modulated_signal = np.concatenate((preamble, modulated_rn16))

        # Interpolate and apply channel effects
        interpolated_signal = self.interpolate_signal(full_modulated_signal)
        if apply_channel_effect:
            return self.channel_effect_strategy.apply(interpolated_signal)

        return interpolated_signal

    def interpolate_signal(self, modulated_signal):
        # Interpolating each bit in the modulated signal based on self.n_samples_TAG_BIT
        # print(f"self.n_samples_TAG_BIT {self.n_samples_TAG_BIT}")
        interpolated_signal = np.repeat(modulated_signal, self.n_samples_TAG_BIT // 2)
        # print(f"len(interpolated_signal) {len(interpolated_signal)}")
        return interpolated_signal


class Reader:
    def __init__(self):
        self.tags = []
        self.received_signals = []

    def add_tag(self, tag):
        self.tags.append(tag)

    def send_query(self, preamble):
        self.received_signals = []
        for tag in self.tags:
            response = tag.respond_to_query(preamble, True)
            if response is not None:
                self.received_signals.append(response)
                self.handle_response(response)

    def handle_response(self, response):
        # Handle the response from tags (e.g., collision detection)
        pass
        # print("Handling tag response...")

    def get_combined_signal(self):
        # Combine all received signals
        if self.received_signals:
            combined_signal = np.sum(self.received_signals, axis=0)
            return combined_signal
        return None


class RFIDSignalSimulator:
    def __init__(self, config: SimulationConfig, noise_ratio=1.0):
        """
        Initialize the simulator with configuration and a noise ratio.

        Args:
            config (SimulationConfig): Configuration containing parameters for simulation.
            noise_ratio (float): Ratio between T1 noise level and RN16/preamble noise level.
        """
        self.sample_rate = config.sample_rate
        self.blf = config.blf
        self.snr = config.snr
        self.preamble = config.preamble
        self.reader = Reader()
        self.noise_ratio = noise_ratio  # Store noise ratio for T1 and RN16 noise levels
        self._initialize_tags(config.tag_configs)

    def _initialize_tags(self, tag_configs):
        for i, tag_config in enumerate(tag_configs):
            modulation_strategy = RFIDFactory.create_modulation('FM0')
            channel_effect_strategy = RFIDFactory.create_channel_effect(
                'Enhanced',
                distance=tag_config.distance,
                snr=tag_config.snr,
                doppler_shift=tag_config.doppler_shift,
                multipath_fading=tag_config.multipath_fading,
                sample_rate=self.sample_rate,
                blf=self.blf,
                num_tags=len(tag_configs),
                phase_rotation=tag_config.phase_rotation,  # Use the phase rotation from tag_config
                attenuation_factor=1.0,  # Customizable attenuation
                max_reflections=1,  # Maximum number of multipath reflections
                reflection_attenuation_range=(0.02, 0.1),  # Range for reflection attenuation
                multipath_delay_range=(0.01, 0.10),  # Range for multipath delay as a fraction of signal length
                phase_shift_factor=2 * np.pi  # Adjust phase shift
            )

            T_READER_FREQ = 40000
            TAG_BIT_D = 1.0 / T_READER_FREQ * pow(10, 6)
            n_samples_TAG_BIT = int(TAG_BIT_D * (self.sample_rate / pow(10, 6)))
            tag = Tag(tag_id=i, modulation_strategy=modulation_strategy,
                      channel_effect_strategy=channel_effect_strategy, n_samples_TAG_BIT=n_samples_TAG_BIT,
                      rn16=tag_config.rn16)
            self.reader.add_tag(tag)

    def simulate(self):
        # Calculate the base noise level
        noise_level = EnhancedChannelEffect.calculate_noise_level(self.snr)

        # Calculate noise levels for T1 and RN16 segments based on noise_ratio
        t1_noise_level = noise_level / self.noise_ratio
        rn16_noise_level = noise_level

        # Generate T1 period (silent period) with modified noise level
        t1_signal = EnhancedChannelEffect.generate_noise(
            int(240 * (self.sample_rate / 1e6)),
            noise_level=t1_noise_level,
            multipath_fading=False
        )

        # Send a query to all tags
        self.reader.send_query(self.preamble)

        # Combine all received signals from tags
        combined_signal = self.reader.get_combined_signal()

        if combined_signal is not None:
            # Generate RN16 noise signal with RN16 noise level and add it to the combined signal
            noise_signal_for_rn16 = EnhancedChannelEffect.generate_noise(
                len(combined_signal),
                noise_level=rn16_noise_level,
                multipath_fading=False
            )
            combined_signal = combined_signal + noise_signal_for_rn16

            # Concatenate T1 and RN16 signals
            combined_signal = np.concatenate((t1_signal, combined_signal))

        return combined_signal

    def plot_results(self, signal):
        # Generate a time vector in microseconds
        time_vector = np.arange(0, len(signal)) / self.sample_rate * 1e6  # Time in microseconds

        # Plot the magnitude of the IQ samples over time
        plt.figure(figsize=(12, 4))
        plt.plot(time_vector, np.abs(signal))
        plt.title("Magnitude of the Received Signal (Including T1 Duration)")
        plt.xlabel("Time [μs]")
        plt.ylabel("Magnitude")
        plt.grid(True)
        plt.show()

        # Plot the time-domain signal (Real part of IQ samples)
        plt.figure(figsize=(12, 4))
        plt.plot(time_vector, signal.real, label='I (In-phase)')
        plt.plot(time_vector, signal.imag, label='Q (Quadrature)', linestyle='--')
        plt.title("Time-Domain Representation of the Received Signal (I and Q Components, Including T1)")
        plt.xlabel("Time [μs]")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot the constellation (IQ plane)
        plt.figure(figsize=(6, 6))
        plt.plot(signal.real, signal.imag, 'o')
        plt.title("IQ Constellation of the Received Signal (Including T1 Duration)")
        plt.xlabel("In-phase (I)")
        plt.ylabel("Quadrature (Q)")
        plt.grid(True)
        plt.show()

    def plot_modulated_signals(self):
        # Create a figure with subplots for each tag
        num_tags = len(self.reader.tags)
        fig, axs = plt.subplots(num_tags, 1, figsize=(10, 2 * num_tags))

        if num_tags == 1:
            axs = [axs]  # Ensure axs is iterable if there's only one tag

        signals = []

        # Collect the modulated signals from each tag
        for tag in self.reader.tags:
            modulated_rn16 = tag.respond_to_query(self.preamble, apply_channel_effect=False)
            signals.append(modulated_rn16)

        combined_signal = np.zeros(len(signals[0]))

        for i, signal in enumerate(signals):
            combined_signal += signal

            # Plot each tag's signal in its subplot using step
            axs[i].step(np.arange(len(signal)), signal, where='mid', label=f'Tag {i + 1}')
            axs[i].set_title(f'Tag {i + 1} Modulated Signal (Preamble + RN16)')
            axs[i].set_ylabel("Amplitude")
            axs[i].grid(True)
            axs[i].legend()

        axs[-1].set_xlabel("Samples")

        # Improve spacing between subplots
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.4)  # Adjust horizontal space between subplots

        # Plot the combined signal in a new figure with different colors using step
        plt.figure(figsize=(12, 4))
        time_vector = np.arange(len(combined_signal))
        for i, signal in enumerate(signals):
            plt.step(time_vector, signal, where='mid', label=f'Tag {i + 1}')

        plt.title("Combined Modulated Signals (Preamble + RN16) from All Tags")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()
        plt.show()

        return combined_signal, signals

    def count_unique_states(self):
        signals = []

        # Collect the modulated signals from each tag
        for tag in self.reader.tags:
            modulated_rn16 = tag.modulation_strategy.modulate(tag.rn16)
            full_signal = np.concatenate((self.preamble, modulated_rn16))
            signals.append(full_signal)

        # Stack the signals vertically and transpose to compare across tags at each time step
        stacked_signals = np.array(signals).T

        # Convert each column in stacked_signals to a tuple and count occurrences of each unique state
        unique_states_dict = {}
        for state in stacked_signals:
            state_tuple = tuple(state)
            if state_tuple in unique_states_dict:
                unique_states_dict[state_tuple] += 1
            else:
                unique_states_dict[state_tuple] = 1

        # Calculate the total number of occurrences to find distribution
        total_occurrences = len(stacked_signals)

        # Calculate the distribution of each unique state
        unique_states_distribution = {state: count / total_occurrences for state, count in unique_states_dict.items()}

        # Number of unique elements
        num_unique_elements = len(unique_states_dict)

        return num_unique_elements, unique_states_dict, unique_states_distribution

    def generate_tag_binaries(self, num_tags, target_unique_states):
        """
        to find if the collided num_tags could generate a target_unique_states or not
        """
        #

        pass


def generate_empty_slot_signal(sample_rate, noise_level=0.1, t1_samples=480, rn16_samples=1100):
    """
    Generates a signal for an empty slot (0 tags) where T1 and RN16 segments have the same noise characteristics.

    Args:
        sample_rate (float): The sampling rate for the signal.
        noise_level (float): Standard deviation of the noise.
        t1_samples (int): Number of samples for the T1 period.
        rn16_samples (int): Number of samples for the RN16 period.

    Returns:
        np.array: Combined complex signal of T1 and RN16 noise.
    """
    # Generate noise for T1 period (same noise level as RN16)
    t1_noise = np.random.normal(0, noise_level, t1_samples) + 1j * np.random.normal(0, noise_level, t1_samples)

    # Generate noise for RN16 period (same noise level as T1)
    rn16_noise = np.random.normal(0, noise_level, rn16_samples) + 1j * np.random.normal(0, noise_level, rn16_samples)

    # Concatenate T1 and RN16 noise to form the full signal
    combined_signal = np.concatenate((t1_noise, rn16_noise))

    return combined_signal


def add_trend(signal, trend_type="random", sine_period_samples=(1500, 4000), min_max_range=(0, 0.5)):
    """
    Adds a trend to both the real and imaginary parts of the signal with random min-max scaling.

    Args:
        signal (np.array): Original complex signal to which the trend will be added.
        trend_type (str): Type of trend to add ("up", "down", "sine", "random").
        sine_period_samples (tuple): Min and max period (in samples) for the sine wave trend.
        min_max_range (tuple): Min and max scaling range for trend amplitude.

    Returns:
        np.array: Signal with the added complex trend.
    """
    # Determine the trend type if set to "random"
    if trend_type == "random":
        trend_type = np.random.choice(["up", "down", "sine"])

    num_samples = len(signal)

    # Randomize the min and max scaling values for the trend amplitude
    trend_min = np.random.uniform(min_max_range[0], min_max_range[1])
    trend_max = np.random.uniform(min_max_range[0], min_max_range[1])

    # Generate the trend
    if trend_type == "up":
        # Linear increasing trend with random scaling
        trend = np.linspace(trend_min, trend_max, num_samples) + 1j * np.linspace(trend_min, trend_max, num_samples)
    elif trend_type == "down":
        # Linear decreasing trend with random scaling
        trend = np.linspace(trend_max, trend_min, num_samples) + 1j * np.linspace(trend_max, trend_min, num_samples)
    elif trend_type == "sine":
        # Select a random period within the given range for a sine wave trend
        period = np.random.randint(sine_period_samples[0], sine_period_samples[1])
        # Generate a sine wave trend with random amplitude scaling
        amplitude = np.random.uniform(trend_min, trend_max)
        trend = amplitude * (np.sin(2 * np.pi * np.arange(num_samples) / period) + 1j * np.sin(
            2 * np.pi * np.arange(num_samples) / period))
    else:
        raise ValueError("Unsupported trend type. Choose 'up', 'down', 'sine', or 'random'.")

    # Apply the trend to the signal by adding it to the original complex signal
    signal_with_trend = signal + trend

    return signal_with_trend


def plot_signal_with_trend(original_signal, signal_with_trend):
    """
    Plots the original signal and the signal with added trend.

    Args:
        original_signal (np.array): Original signal without trend.
        signal_with_trend (np.array): Signal after adding the trend.
    """
    num_samples = len(original_signal)
    time_axis = np.arange(num_samples)

    # Plot original signal
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, original_signal.real, label="I (Original)", color="blue")
    plt.plot(time_axis, original_signal.imag, label="Q (Original)", color="orange")
    plt.title("Original Signal")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    # Plot signal with trend
    plt.subplot(2, 1, 2)
    plt.plot(time_axis, signal_with_trend.real, label="I (With Trend)", color="blue", linestyle="--")
    plt.plot(time_axis, signal_with_trend.imag, label="Q (With Trend)", color="orange", linestyle="--")
    plt.title("Signal with Added Trend")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def generate_collision_dataset(
        num_samples_per_class=20,
        num_classes=[0, 1, 2, 3, 4],
        sample_rate=2e6,
        noise_level=0.1,
        preamble=np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1]),
        filename="collision_database.csv",
        snr_range=(30, 50),
        ambiguous_snr_range=(0, 5),
        high_tag_count_range=(20, 100)
):
    """
    Generates a collision dataset by simulating RFID tag responses under various conditions.

    Args:
        num_samples_per_class (int): Number of samples per class to generate.
        num_classes (list): List of different tag counts to simulate, e.g., [0, 1, 2, 3, 4].
        sample_rate (float): Sampling rate in Hz.
        noise_level (float): Noise level for empty slot simulation.
        preamble (np.array): Modulated preamble for RFID tags.
        filename (str): Name of the CSV file to save the dataset.
        snr_range (tuple): SNR range for regular cases with defined tag counts.
        ambiguous_snr_range (tuple): SNR range for ambiguous/unclassifiable signal cases.
        high_tag_count_range (tuple): Range for random number of tags in ambiguous cases.

    Returns:
        None. Saves the generated dataset to a CSV file.
    """
    # Lists to store the generated data for each feature and label
    all_I_values = []
    all_Q_values = []
    all_Phase_values = []
    all_Abs_values = []
    all_labels = []

    for num_tags in num_classes:
        for sample_idx in range(num_samples_per_class):
            # Case for 0 tags (empty slot)
            if num_tags == 0:
                combined_signal = generate_empty_slot_signal(
                    sample_rate=sample_rate, noise_level=noise_level, t1_samples=480, rn16_samples=1100
                )
                label = 0  # Label for empty slot

            # Case for an ambiguous or unclassifiable signal (tag count 4 or other specific conditions)
            elif num_tags == 4:
                combined_signal, label = generate_ambiguous_signal(
                    sample_rate, preamble, ambiguous_snr_range, high_tag_count_range
                )

            # Case for regular defined tag counts
            else:
                combined_signal, label = generate_defined_tag_signal(
                    sample_rate, num_tags, preamble, snr_range
                )

            # Continue if the signal is not valid
            if combined_signal is None:
                print("Warning: Combined signal is None. Skipping.")
                continue

            # Process and add trend to the signal
            combined_signal = combined_signal[180:]  # Trim initial samples
            signal_with_trend = add_trend(combined_signal, trend_type="random")

            # Extract features
            I_values = signal_with_trend.real.tolist()
            Q_values = signal_with_trend.imag.tolist()
            Phase_values = np.angle(signal_with_trend).tolist()
            Abs_values = np.abs(signal_with_trend).tolist()

            # Append features and label to lists
            all_I_values.append(I_values)
            all_Q_values.append(Q_values)
            all_Phase_values.append(Phase_values)
            all_Abs_values.append(Abs_values)
            all_labels.append(label)

    # Create and save DataFrame
    collision_database = pd.DataFrame({
        'I_values': all_I_values,
        'Q_values': all_Q_values,
        'Phase_values': all_Phase_values,
        'Abs_values': all_Abs_values,
        'Label': all_labels
    })
    collision_database.to_csv(filename, index=False)
    print(f"Data has been saved to '{filename}'.")
    print(f"The CSV file contains {collision_database.shape[0]} rows and {collision_database.shape[1]} columns.")


def generate_ambiguous_signal(sample_rate, preamble, ambiguous_snr_range, high_tag_count_range):
    """Generate a signal for a case with ambiguous/unclassifiable signals."""
    high_tag_count = random.randint(*high_tag_count_range)
    low_snr = random.uniform(*ambiguous_snr_range)

    # Create tag configurations with random parameters
    tag_configs = [
        TagConfig(
            distance=random.uniform(1, 3),
            snr=low_snr,
            doppler_shift=random.uniform(-150, 150),
            multipath_fading=random.choice([True, False]),
            phase_rotation=random.uniform(0, 360),
            rn16=np.random.randint(0, 2, 16)
        )
        for _ in range(high_tag_count)
    ]

    # Create and run simulator
    sim_config = SimulationConfig(
        sample_rate=sample_rate,
        blf=40e3,
        snr=low_snr,
        tag_configs=tag_configs,
        preamble=preamble
    )
    simulator = RFIDSignalSimulator(sim_config, noise_ratio=4.0)
    combined_signal = simulator.simulate()
    return combined_signal, 4  # Return signal and label for ambiguous case


def generate_defined_tag_signal(sample_rate, num_tags, preamble, snr_range):
    """Generate a signal for a defined number of tags (1, 2, etc.)."""
    tag_configs = [
        TagConfig(
            distance=random.uniform(0.1, 3.0),
            snr=random.uniform(*snr_range),
            doppler_shift=random.uniform(-150, 150),
            multipath_fading=random.choice([True, False]),
            phase_rotation=random.uniform(0, 360),
            rn16=np.random.randint(0, 2, 16)
        )
        for _ in range(num_tags)
    ]

    # Create and run simulator
    sim_config = SimulationConfig(
        sample_rate=sample_rate,
        blf=40e3,
        snr=random.uniform(*snr_range),
        tag_configs=tag_configs,
        preamble=preamble
    )
    simulator = RFIDSignalSimulator(sim_config)
    combined_signal = simulator.simulate()
    return combined_signal, num_tags  # Return signal and label for defined tag count


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
    possible_rn16_values = list(itertools.product([0, 1], repeat=3))  # All 16-bit binary sequences

    # Iterate through each permutation of RN16 sequences for the specified number of tags
    for rn16_permutation in itertools.permutations(possible_rn16_values, num_tags):
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

        # Update the count for this unique state value
        unique_state_counts[num_unique_elements] += 1

    return dict(unique_state_counts)

########################################################################################################################
##################### usage example

# sample_rate = 2e6  # 4 MHz sampling rate
# blf = 40e3  # 40 kHz BLF
# snr = 20
#
# tag_configs = [
#     TagConfig(distance=0.7, snr=20, doppler_shift=10, multipath_fading=True, phase_rotation=0,  rn16=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1])),
#     TagConfig(distance=0.8, snr=20, doppler_shift=10, multipath_fading=True, phase_rotation=90, rn16=np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0])),
#     TagConfig(distance=0.9, snr=20, doppler_shift=10, multipath_fading=True, phase_rotation=45, rn16=np.array([1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1])),
#     TagConfig(distance=1.0, snr=20, doppler_shift=0, multipath_fading=False, phase_rotation=70, rn16=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])),
#     TagConfig(distance=1.0, snr=20, doppler_shift=0, multipath_fading=False, phase_rotation=70, rn16=np.array([1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0])),
# ]
# # Define the preamble (already modulated)
# modulated_preamble = np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1])  # Example preamble bits already modulated
#
# # Create a SimulationConfig
#
# sim_config = SimulationConfig(
#     sample_rate=sample_rate,
#     blf=blf,
#     snr=snr,
#     tag_configs=tag_configs,
#     preamble=modulated_preamble
# )
#
# # Initialize and run the simulator
# simulator = RFIDSignalSimulator(sim_config)
# collied_signal = simulator.simulate()
# print(f" len of collied_signal {len(collied_signal)}")


