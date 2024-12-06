import numpy as np
import pandas as pd
import streamlit as st


class ParticleFilter:
    """
    A particle filter for estimating parameters using a set of particles and resampling
    based on weights derived from observations.

    :param num_particles: Number of particles to use in the filter.
    :param process_noise: Process noise standard deviation for each parameter [beta0, beta1, beta2].
    :param measurement_noise: Measurement noise standard deviation for observations.
    """

    def __init__(self, num_particles=1000, process_noise=None, measurement_noise=0.05):
        self.num_particles = num_particles
        self.process_noise = np.array(process_noise or [0.01, 0.001, 0.01])
        self.measurement_noise = measurement_noise

    def initialize_particles(self, beta0_range, beta1_range, beta2_range):
        """
        Initialize particles within the given parameter ranges.

        :param beta0_range: Tuple specifying the range for beta0.
        :param beta1_range: Tuple specifying the range for beta1.
        :param beta2_range: Tuple specifying the range for beta2.
        :return: A numpy array of initialized particles with shape (num_particles, 3).
        """
        particles = np.random.uniform(
            [beta0_range[0], beta1_range[0], beta2_range[0]],
            [beta0_range[1], beta1_range[1], beta2_range[1]],
            (self.num_particles, 3)
        )
        return particles

    def propagate_particles(self, particles):
        """
        Add process noise to particles to simulate system dynamics.

        :param particles: A numpy array of particles with shape (num_particles, 3).
        :return: Updated particles with added noise.
        """
        noise = np.random.normal(0, self.process_noise, particles.shape)
        particles += noise
        return particles

    def compute_weights(self, particles, observation, time):
        """
        Compute weights for each particle based on the likelihood of observing the given data.

        :param particles: A numpy array of particles with shape (num_particles, 3).
        :param observation: The observed data point (length measured).
        :param time: Time in months for the observation.
        :return: Normalized weights for each particle.
        """
        beta0, beta1, beta2 = particles[:, 0], particles[:, 1], particles[:, 2]
        predicted_observations = beta2 / (1 + np.exp(-(beta0 + beta1 * time)))

        weights = np.exp(-0.5 * ((observation - predicted_observations) ** 2) / self.measurement_noise ** 2)
        weights /= np.sum(weights)  # Normalize weights
        return weights

    def resample_particles(self, particles, weights):
        """
        Resample particles based on their weights using systematic resampling.

        :param particles: A numpy array of particles with shape (num_particles, 3).
        :param weights: Normalized weights for each particle.
        :return: Resampled particles with shape (num_particles, 3).
        """
        indices = np.random.choice(self.num_particles, size=self.num_particles, p=weights)
        return particles[indices]

    def filter(self, df, beta0_range, beta1_range, beta2_range):
        """
        Perform particle filtering on a DataFrame of observations.

        :param df: A DataFrame containing 'item_id', 'time (months)', and 'length_measured' columns.
        :param beta0_range: Tuple specifying the range for beta0.
        :param beta1_range: Tuple specifying the range for beta1.
        :param beta2_range: Tuple specifying the range for beta2.
        :return: A DataFrame with filtered results including estimated parameters and predicted crack lengths.
        """
        filtered_data = []
        num_items = len(df['item_id'].unique())
        progress_bar = st.progress(0)

        with st.spinner('Particles filtering...'):
            for i, item_index in enumerate(df['item_id'].unique()):
                df_item = df[df['item_id'] == item_index]
                particles = self.initialize_particles(beta0_range, beta1_range, beta2_range)

                for _, row in df_item.iterrows():
                    time = row['time (months)']
                    observation = row['length_measured']

                    particles = self.propagate_particles(particles)                 # Particle propagation, weight computation, and resampling
                    weights = self.compute_weights(particles, observation, time)
                    particles = self.resample_particles(particles, weights)


                    estimated_state = np.mean(particles, axis=0)                    # Estimate the mean state and crack length
                    estimated_crack_length = estimated_state[2] / (
                                1 + np.exp(-(estimated_state[0] + estimated_state[1] * time)))

                    filtered_data.append(row.tolist() + [estimated_crack_length] + estimated_state.tolist())        # Append filtered data

                progress = (i + 1) / num_items      # Update progress bar
                progress_bar.progress(progress)

        column_names = df.columns.tolist() + ['length_filtered', 'beta0', 'beta1', 'beta2']
        filtered_df = pd.DataFrame(filtered_data, columns=column_names)

        return filtered_df
