import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    roc_curve, auc, precision_recall_curve
)

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def calculate_combined_metrics(all_y_true, all_y_pred):
    """
    Calculates combined metrics from all true labels and predictions.
    Args:
        all_y_true (np.ndarray): Array of combined true labels.
        all_y_pred (np.ndarray): Array of combined predictions.
    Returns:
        dict: Dictionary containing the global metrics.
    """
    metrics = {
        'precision_weighted': precision_score(all_y_true, all_y_pred, average='weighted'),
        'recall_weighted': recall_score(all_y_true, all_y_pred, average='weighted'),
        'f1_weighted': f1_score(all_y_true, all_y_pred, average='weighted'),
        'accuracy': accuracy_score(all_y_true, all_y_pred),
        'precision_macro': precision_score(all_y_true, all_y_pred, average='macro'),
        'recall_macro': recall_score(all_y_true, all_y_pred, average='macro'),
        'f1_macro': f1_score(all_y_true, all_y_pred, average='macro'),
    }
    return metrics


def measure_performance_and_plot(true_labels, predicted_labels):
    """
    Measures model performance and plots relevant metrics.
    """
    # Classification Report
    report = classification_report(true_labels, predicted_labels, output_dict=True)
    print(f"-- Global Classification Report --\n" + '_' * 45)
    st.dataframe(pd.DataFrame(report).transpose())
    print(pd.DataFrame(report).transpose())
    print("\n")

    # Combined Metrics
    combined_metrics = calculate_combined_metrics(true_labels, predicted_labels)
    print("\n-- Global Metrics --\n" + '_' * 45)
    for metric_name, value in combined_metrics.items():
        print(f"- {metric_name}: {value:.4f}")

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
        fpr, tpr, _ = roc_curve(true_labels, predicted_labels)
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
        precision, recall, _ = precision_recall_curve(true_labels, predicted_labels)
        plt.figure()
        plt.plot(recall, precision, color='b', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        st.pyplot(plt)

