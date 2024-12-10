import streamlit as st
import os
import warnings
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import visualkeras
from tensorflow.keras.utils import plot_model

from src.forecasting.validation.display import DisplayData

warnings.filterwarnings("ignore", message=".*legend_text_spacing_offset.*")


def save_predictions(output_path, df, step):
    file_path = f"{output_path}/lstm_predictions_{step}.csv"
    df.to_csv(file_path, index=False)
    return f"Predictions saved successfully: {output_path}"


def generate_model_images(model, MODEL_FOLDER):
    """
    Generates and displays images of the model architecture and its stylized version.
    Parameters:
    - model : The Keras model to visualize.
    - MODEL_FOLDER : The folder where the images will be saved.
    """
    try:
        os.makedirs(MODEL_FOLDER, exist_ok=True)

        plot_model(model, to_file=f'{MODEL_FOLDER}architecture.png', show_shapes=True, show_layer_names=True)

        stylized_img = visualkeras.layered_view(model, legend=True)
        stylized_img.save(f'{MODEL_FOLDER}model_stylized.png')

        fig, axs = plt.subplots(1, 2, figsize=(15, 20))

        img1 = mpimg.imread(f'{MODEL_FOLDER}architecture.png')      # Display the architecture image
        axs[0].imshow(img1)
        axs[0].axis('off')
        axs[0].set_title('Architecture')

        img2 = mpimg.imread(f'{MODEL_FOLDER}model_stylized.png')    # Display the stylized model image
        axs[1].imshow(img2)
        axs[1].axis('off')
        axs[1].set_title('Stylized Model')

        plt.tight_layout()
        fig.savefig(f'{MODEL_FOLDER}/model_visualization.png', bbox_inches='tight', dpi=300)
        #st.pyplot(fig)

    except Exception as e:
        st.error('Error while generating model architecture images. '
                 'You must install graphviz and pydot to generate the model architecture images.')


def display_results(df):
    display = DisplayData(df)

    col1, col2 = st.columns(2)
    with col1:
        display.plot_discrete_scatter(df, 'time (months)', 'length_measured', 'source')
        display.plot_discrete_scatter(df, 'time (months)', 'length_measured', 'item_id')
        display.plot_discrete_scatter(df, 'time (months)', 'length_measured', 'Fatigue crack')
    with col2:
        display.plot_discrete_scatter(df, 'time (months)', 'length_filtered', 'source')
        display.plot_discrete_scatter(df, 'time (months)', 'length_filtered', 'item_id')
        display.plot_discrete_scatter(df, 'time (months)', 'length_filtered', 'Fatigue crack')

    display.plot_discrete_scatter(df, 'time (months)', 'length_filtered', 'item_id')
    st.dataframe(df)

    col1, col2 = st.columns([2, 2])

    qualitative_pal = 'Safe'

    with col1:
        display.plot_scatter_with_color(df=df,
                            x_col='time (months)',
                            y_col='length_measured',
                            color_col='Failure mode',
                            palette_type='qualitative',
                            palette_name=qualitative_pal)

    with col2:
        display.plot_scatter_with_color(df=df,
                            x_col='time (months)',
                            y_col='length_filtered',
                            color_col='Failure mode',
                            palette_type='qualitative',
                            palette_name=qualitative_pal)

    col1, col2 = st.columns(2)
    with col1:
        display.plot_histogram_with_color(df=df,
                            x_col='time (months)',
                            y_col='length_measured',
                            color_col='Failure mode',
                            palette_type='qualitative',
                            palette_name=qualitative_pal)
    with col2:
        display.plot_histogram_with_color(df=df,
                            x_col='time (months)',
                            y_col='length_filtered',
                            color_col='Failure mode',
                            palette_type='qualitative',
                            palette_name=qualitative_pal)
