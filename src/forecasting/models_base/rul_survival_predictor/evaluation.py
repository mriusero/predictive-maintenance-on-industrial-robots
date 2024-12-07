import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    roc_curve, auc, precision_recall_curve
)


def measure_performance_and_plot(predictions_df: pd.DataFrame):
    """
    Measures model performance and plots relevant metrics.
    """
    true_labels = predictions_df['label_y']
    predicted_labels = predictions_df['predicted_failure_6_months_binary']

    # Classification Report
    report = classification_report(true_labels, predicted_labels, output_dict=True)
    st.write("Classification Report:")
    st.dataframe(pd.DataFrame(report).transpose())
    print(pd.DataFrame(report).transpose())
    print("\n")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Confusion Matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
        cm_display.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        st.pyplot(plt)

    with col2:
        # ROC Curve
        fpr, tpr, _ = roc_curve(true_labels, predictions_df['predicted_failure_6_months'])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        st.pyplot(plt)

    with col3:
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(true_labels, predictions_df['predicted_failure_6_months'])
        plt.figure()
        plt.plot(recall, precision, color='b', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        st.pyplot(plt)


def display_results(x_train, predictions_df):
    """
    Displays results in Streamlit.
    """
    col1, col2 = st.columns(2)
    with col1:
        st.write(x_train.columns.to_list())

    measure_performance_and_plot(predictions_df)