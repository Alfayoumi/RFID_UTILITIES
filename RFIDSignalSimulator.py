import itertools

import numpy as np
from matplotlib import pyplot as plt


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
        print("Handling tag response...")

    def get_combined_signal(self):
        # Combine all received signals
        if self.received_signals:
            combined_signal = np.sum(self.received_signals, axis=0)
            return combined_signal
        return None


# Main Simulator using the patterns
class RFIDSignalSimulator:
    def __init__(self, config: SimulationConfig):
        self.sample_rate = config.sample_rate
        self.blf = config.blf
        self.snr = config.snr
        self.preamble = config.preamble
        self.reader = Reader()
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
        # T1 period (silent period) simulation
        noise_level = EnhancedChannelEffect.calculate_noise_level(self.snr)
        t1_signal = EnhancedChannelEffect.generate_noise(int(240 * (self.sample_rate / 1e6)), noise_level=noise_level,
                                                         multipath_fading=False)

        # Send a query to all tags
        self.reader.send_query(self.preamble)

        # Combine all received signals
        combined_signal = self.reader.get_combined_signal()

        if combined_signal is not None:
            # Generate a noise signal with the same distribution as T1 and add it to the combined signal
            noise_signal_for_rn16 = EnhancedChannelEffect.generate_noise(len(combined_signal), noise_level=noise_level,
                                                                         multipath_fading=False)
            combined_signal = combined_signal + noise_signal_for_rn16
            combined_signal = np.concatenate((t1_signal, combined_signal))
            self.plot_results(combined_signal)
            # self.plot_results(combined_signal[480:-1])
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
        Efficiently generate a list of unique binary sequences for a given number of tags using on-the-fly permutations.

        Args:
        - num_tags (int): The number of tags, which should be greater than 2 and smaller than 65536.
        - target_unique_states (int): The target number of unique modulation states, which should be <= 2^num_tags.

        Returns:
        - List of binary sequences where each row represents a 16-bit binary list.
        """
        # Validate input arguments
        if num_tags <= 2 or num_tags >= 65536:
            raise ValueError("num_tags must be greater than 2 and smaller than 65536")
        if target_unique_states > 2 ** num_tags:
            raise ValueError(f"target_unique_states cannot be greater than 2^{num_tags} ({2 ** num_tags})")

        # Generator for all possible binary sequences (lazy evaluation)
        binary_sequences = itertools.product([0, 1], repeat=16)  # Generate all possible 16-bit binary sequences

        # Initialize a list to store the selected sequences
        selected_sequences = []

        # Generate and test sequences on the fly
        for binary_sequence in binary_sequences:
            selected_sequences.append(binary_sequence)  # Add the sequence to the list

            # Stop when we have enough sequences (equal to num_tags)
            if len(selected_sequences) == num_tags:
                break

        # Assign the generated RN16 sequences to each tag in the simulator's tags
        for tag, binary_sequence in zip(self.reader.tags, selected_sequences):
            tag.rn16 = np.array(binary_sequence)

        # Check how many unique states are present in the modulation
        num_unique_states, _, _ = self.count_unique_states()

        # If the number of unique states matches the target, return the binary sequences
        if num_unique_states == target_unique_states:
            return selected_sequences
        else:
            # If the number of unique states doesn't match, retry (though this should rarely happen)
            return self.generate_tag_binaries(num_tags, target_unique_states)


sample_rate = 2e6  # 4 MHz sampling rate
blf = 40e3  # 40 kHz BLF
snr = 20

tag_configs = [
    TagConfig(distance=0.7, snr=20, doppler_shift=10, multipath_fading=True, phase_rotation=0,  rn16=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1])),
    TagConfig(distance=0.8, snr=20, doppler_shift=10, multipath_fading=True, phase_rotation=90, rn16=np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0])),
    TagConfig(distance=0.9, snr=20, doppler_shift=10, multipath_fading=True, phase_rotation=45, rn16=np.array([1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1])),
#    TagConfig(distance=1.0, snr=20, doppler_shift=0, multipath_fading=False, phase_rotation=70, rn16=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])),
#    TagConfig(distance=1.0, snr=20, doppler_shift=0, multipath_fading=False, phase_rotation=70, rn16=np.array([1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0])),
]
# Define the preamble (already modulated)
modulated_preamble = np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1])  # Example preamble bits already modulated

# Create a SimulationConfig
sim_config = SimulationConfig(
    sample_rate=sample_rate,
    blf=blf,
    snr=snr,
    tag_configs=tag_configs,
    preamble=modulated_preamble
)

# Initialize and run the simulator
simulator = RFIDSignalSimulator(sim_config)
collied_signal = simulator.simulate()
print(f" len of collied_signal {len(collied_signal)}")


