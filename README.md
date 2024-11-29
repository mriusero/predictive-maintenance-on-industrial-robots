# Predictive Maintenance for Industrial Robots

This project focuses on predictive maintenance for industrial robots engaged in nuclear fuel replacement tasks. It combines data analysis, machine learning, and decision-making frameworks to improve fleet management and operational efficiency.

---

## Table of Contents
1. [Installation](#installation)
2. [Running the Application](#running-the-application)
    - [Running Locally](#running-locally)
    - [Running with Docker](#running-with-docker)
3. [Project Overview](#project-overview)
4. [Project Phases](#project-phases)
5. [Technical Phases](#technical-phases)
6. [Detailed Phases](#detailed-phases)
7. [Data Structure](#data-structure)
8. [Directory Structure](#input-data-structure)
9. [Lessons Learned](#lessons-learned)

---

## Installation

### Prerequisites
- **Python**: Version 3.8 or higher  
- **pip**: Python package installer  

### Clone the Repository
```bash
git clone https://github.com/mriusero/predictive-maintenance-on-industrial-robots
cd predictive-maintenance-on-industrial-robots
```

### Running the Application

You can run the application either locally or using Docker.

#### Running Locally

1. **Create and Activate a Virtual Environment**:
   - **macOS/Linux**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - **Windows**:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```

2. **Install Required Packages**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Launch the Application**:
   ```bash
   streamlit run app.py
   ```

#### Running with Docker

1. **Build the Docker Image**:
   ```bash
   docker build -t streamlit .
   ```

2. **Run the Docker Container**:
   ```bash
   docker run -p 8501:8501 streamlit
   ```

Access the application at [http://localhost:8501](http://localhost:8501).

---

## Project Overview

The objective of this project is to enhance predictive maintenance practices for industrial robots, minimizing downtime and ensuring operational continuity. The focus is on leveraging failure data and machine learning models to predict robot performance and make informed maintenance decisions.

---

## Project Phases

1. **Context**: Introduction to predictive maintenance and its importance.
2. **Environment**: Description of the technical setup and dependencies.
3. **Knowledge Base**: Key mathematical and machine learning concepts.
4. **Input Data**: Structure and format of data used in the project.
5. **Testing**: Methods for evaluating model performance.
6. **Exploration**: Initial data analysis and feature engineering.

---

## Technical Phases

- **Feature Engineering**: Extraction of time series and statistical features from failure data.
- **Prediction Models**: Use of models such as Random Forest, LSTM, and Gradient Boosting Survival Models for Remaining Useful Life (RUL) predictions.
- **Performance Evaluation**: Metrics include accuracy, confusion matrix, ROC-AUC, and precision-recall curves.
- **Remaining Work**: Development of a decision-making model for Phase II.
- **Lessons Learned**: Insights on project management, version control, and team collaboration.

---

## Detailed Phases

### Phase I – Remaining Useful Life (RUL) Prediction
Objective: Predict if a robot will remain operational for the next six months based on historical failure data.

- **Evaluation Criteria**: Rewards and penalties are assigned based on the accuracy of RUL predictions.

### Phase II – Maintenance Decision-Making
Objective: Decide whether to replace robots before missions, ensuring a minimum of 10 operational robots.

- **Evaluation Criteria**: Rewards for successful missions, penalties for failures or unnecessary replacements.

---

## Data Structure

- **`failure_data.csv`**: Summary of failure modes and timestamps.
- **`Degradation data`**: Time series data on crack length evolution.
- **`Testing Data`**: Datasets for model validation based on real scenarios.

---

## Input Data Structure
```plaintext
.
├── testing_data
│   └── sent_to_student
│       ├── group_0
│       │   ├── Sample_submission.csv
│       │   ├── testing_data.rar
│       │   └── testing_item_(n).csv  // n = 50 files
│       ├── scenario_0
│       │   └── item_(n).csv        // n = 11 files
│       ├── scenario_1
│       │   └── item_(n).csv        // n = 11 files
│       ├── scenario_2
│       │   └── item_(n).csv        // n = 11 files
│       ├── scenario_3
│       │   └── item_(n).csv        // n = 11 files
│       ├── scenario_4
│       │   └── item_(n).csv        // n = 11 files
│       ├── scenario_5
│       │   └── item_(n).csv        // n = 11 files
│       ├── scenario_6
│       │   └── item_(n).csv        // n = 11 files
│       ├── scenario_7
│       │   └── item_(n).csv        // n = 11 files
│       ├── scenario_8
│       │   └── item_(n).csv        // n = 11 files
│       ├── scenario_9
│       │   └── item_(n).csv        // n = 11 files
│       └── testing_data_phase_2.rar
│
└── training_data
    ├── create_a_pseudo_testing_dataset.ipynb
    ├── degradation_data
    │   └── item_(n).csv            // n = 50 files
    ├── failure_data.csv
    ├── pseudo_testing_data
    │   └── item_(n).csv            // n = 50 files
    └── pseudo_testing_data_with_truth
        ├── Solution.csv
        └── item_(n).csv            // n = 50 files
```

---

## Lessons Learned

- **Model Selection**: Importance of choosing models suited to time-series data.
- **Project Organization**: Emphasis on setting time-focused goals and using version control.
- **Collaboration**: Benefits of a well-structured repository and documentation for teamwork.