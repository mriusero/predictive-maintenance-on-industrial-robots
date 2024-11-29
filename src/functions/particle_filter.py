import numpy as np
import pandas as pd
import streamlit as st

class ParticleFilter:
    def __init__(self, num_particles=1000, process_noise=None, measurement_noise=0.05):
        self.num_particles = num_particles
        self.process_noise = np.array(process_noise or [0.01, 0.001, 0.01])
        self.measurement_noise = measurement_noise

    def initialize_particles(self, beta0_range, beta1_range, beta2_range):
        # Utilisation de np.random.uniform avec (N, 3) pour générer directement les particules
        particles = np.random.uniform(
            [beta0_range[0], beta1_range[0], beta2_range[0]],
            [beta0_range[1], beta1_range[1], beta2_range[1]],
            (self.num_particles, 3)
        )
        return particles

    def propagate_particles(self, particles):
        # Utilisation de broadcasting pour ajouter du bruit de processus
        noise = np.random.normal(0, self.process_noise, particles.shape)
        particles += noise
        return particles

    def compute_weights(self, particles, observation, time):
        # Calculs vectorisés pour prédire les observations et les poids
        beta0, beta1, beta2 = particles[:, 0], particles[:, 1], particles[:, 2]
        predicted_observations = beta2 / (1 + np.exp(-(beta0 + beta1 * time)))
        # Calcul vectorisé des poids
        weights = np.exp(-0.5 * ((observation - predicted_observations) ** 2) / self.measurement_noise ** 2)
        weights /= np.sum(weights)  # Normalisation des poids
        return weights

    def resample_particles(self, particles, weights):
        # Résample les particules en utilisant numpy directement
        indices = np.random.choice(self.num_particles, size=self.num_particles, p=weights)
        return particles[indices]

    def filter(self, df, beta0_range, beta1_range, beta2_range):
        filtered_data = []
        num_items = len(df['item_id'].unique())
        progress_bar = st.progress(0)

        with st.spinner('Particles filtering...'):
            for i, item_index in enumerate(df['item_id'].unique()):
                df_item = df[df['item_id'] == item_index]
                particles = self.initialize_particles(beta0_range, beta1_range, beta2_range)

                # Utilisation de df.iterrows est conservée car chaque ligne est nécessaire pour des calculs individuels
                for index, row in df_item.iterrows():
                    time = row['time (months)']
                    observation = row['length_measured']

                    # Propagation, pondération et rééchantillonnage des particules
                    particles = self.propagate_particles(particles)
                    weights = self.compute_weights(particles, observation, time)
                    particles = self.resample_particles(particles, weights)

                    # Estimation de l'état moyen
                    estimated_state = np.mean(particles, axis=0)
                    estimated_crack_length = estimated_state[2] / (1 + np.exp(-(estimated_state[0] + estimated_state[1] * time)))

                    # Ajouter les données filtrées à la liste
                    filtered_data.append(row.tolist() + [estimated_crack_length] + estimated_state.tolist())

                # Mise à jour de la barre de progression
                progress = (i + 1) / num_items
                progress_bar.progress(progress)

        # Création du DataFrame avec les résultats filtrés
        column_names = df.columns.tolist() + ['length_filtered', 'beta0', 'beta1', 'beta2']
        filtered_df = pd.DataFrame(filtered_data, columns=column_names)

        return filtered_df