import json

import numpy as np
import pandas as pd
import streamlit as st

from src.forecasting.preprocessing.features import FeatureAdder
from .configs import MODEL_FOLDER, MIN_SEQUENCE_LENGTH, FEATURE_COLUMNS, TARGET_COLUMNS


def extract_features_for_item(item_data, feature_columns):
    features = {}
    for col in feature_columns:
        features[col] = item_data[col].values
    return features


def extract_targets_for_item(item_data, target_columns, start_index, end_index):
    targets = {}
    for col in target_columns:
        targets[col] = item_data[col].values[start_index:end_index]
    return targets


def prepare_initial_data(item_data, item_index, feature_columns=None, target_columns=None, start_index=None, end_index=None):
    """
    Prépare les données initiales pour un item spécifique en extrayant les caractéristiques et les cibles.
    Paramètres :
        item_data : pd.DataFrame, données pour l'item spécifique.
        item_index : int, identifiant de l'item.
        source : int, la source des données (0 pour les données initiales, 1 pour les prévisions).
        feature_columns : list de str, les colonnes de caractéristiques à ajouter.
        target_columns : list de str, les colonnes de cibles à ajouter.
        start_index : int, index de début pour l'extraction des cibles.
        end_index : int, index de fin pour l'extraction des cibles.
    Retourne :
        pd.DataFrame avec les données de l'item, les caractéristiques extraites et les cibles.
    """
    times = item_data['time (months)'].values
    features = extract_features_for_item(item_data, feature_columns)
    data_dict = {
        'item_id': item_index,
        'time (months)': times,
        'source': 0
    }
    data_dict.update(features)
    return pd.DataFrame(data_dict)


def add_predictions_to_data(initial_data, min_sequence_length=MIN_SEQUENCE_LENGTH, feature_columns=FEATURE_COLUMNS, target_columns=TARGET_COLUMNS):
    """
    Ajoute les prévisions aux données et prépare les données étendues pour chaque item.
    Paramètres :
        df : pd.DataFrame, les données d'entrée.
        predictions : list de dict, prévisions pour chaque item.
        min_sequence_length : int, longueur minimale des séquences pour l'entraînement.
        feature_columns : list de str, les colonnes de caractéristiques à inclure dans les séquences.
        target_columns : list de str, les colonnes de cibles à inclure dans les séquences.
    Retourne :
        pd.DataFrame avec les données étendues, y compris les prévisions.
    """
    df = initial_data.copy()
    predictions = json.load(open(f'{MODEL_FOLDER}' + 'predictions.json', 'r'))

    if feature_columns is None or target_columns is None:
        raise ValueError("feature_columns et target_columns doivent être fournis.")

    item_indices = df['item_id'].unique()
    extended_data = []

    for idx, item_index in enumerate(item_indices):
        item_data = df[df['item_id'] == item_index].sort_values(by='time (months)')
        max_time = np.max(item_data['time (months)'].values)

        pred = predictions[idx]

        required_keys = [
            "length_filtered", "length_measured",
            "Infant mortality", "Control board failure", "Fatigue crack"
        ]
        if not all(key in pred for key in required_keys):
            raise KeyError(f"Les clés requises sont manquantes dans les prédictions pour l'item_id {item_index}.")

        future_lengths_filtered = np.array(pred["length_filtered"])
        future_lengths_measured = np.array(pred["length_measured"])
        classified_infant_mortality = np.array(pred["Infant mortality"])
        classified_control_board_failure = np.array(pred["Control board failure"])
        classified_fatigue_crack = np.array(pred["Fatigue crack"])

        forecast_length = len(future_lengths_filtered)
        future_times = np.arange(np.ceil(max_time + 1), np.ceil(max_time + 1) + forecast_length)

        start_index = max(0, len(item_data) - min_sequence_length)  # Éviter un indice négatif
        end_index = len(item_data)

        initial_data = prepare_initial_data(
            item_data, item_index,
            feature_columns=feature_columns,
            target_columns=target_columns,
            start_index=start_index,
            end_index=end_index
        )
        forecast_data = pd.DataFrame({
            'item_id': item_index,
            'time (months)': future_times,
            'length_filtered': future_lengths_filtered,
            'length_measured': future_lengths_measured,
            'Infant mortality': classified_infant_mortality,
            'Control board failure': classified_control_board_failure,
            'Fatigue crack': classified_fatigue_crack,
            'source': 1
        })
        extended_data.append(pd.concat([initial_data, forecast_data]))

    if not extended_data:
        raise ValueError("Aucune donnée étendue n'a été créée avec les prévisions fournies.")

    df_extended = pd.concat(extended_data).reset_index(drop=True)

    feature_adder = FeatureAdder(min_sequence_length=min_sequence_length)
    df_extended = feature_adder.add_features(df_extended, particles_filtery=False)

    # Verification
    columns_to_keep = [
        'item_id',
        'time (months)',
        'length_filtered',
        'length_measured',
        'Infant mortality',
        'Control board failure',
        'Fatigue crack',
        'source'
    ]
    df_filtered = df_extended[columns_to_keep]
    df_sorted = df_filtered.sort_values(by=['item_id', 'time (months)'])
    st.dataframe(df_sorted)

    return df_extended
