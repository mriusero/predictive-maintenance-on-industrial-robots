# Predictive Maintenance for Industrial Robots

This project explores predictive maintenance strategies for industrial robots used in nuclear fuel replacement tasks. It leverages data analysis, machine learning models, and decision-making frameworks to enhance fleet management and maximize operational efficiency.

**Project Phases:**
1. **Context:** Overview of predictive maintenance requirements.
2. **Environment:** Description of the technical environment and dependencies.
3. **Knowledge Base:** Core mathematical and machine learning concepts.
4. **Input Data:** Details on data structure and file formats.
5. **Testing:** Model performance evaluation process.
6. **Exploration:** Initial data analysis and engineering techniques.

**Technical Phases:**
- **Feature Engineering:** Time series and statistical features derived from failure data.
- **Prediction Models:** Various models such as Random Forest, LSTM, and Gradient Boosting Survival Models for Remaining Useful Life (RUL) predictions.
- **Performance Evaluation:** Metrics including accuracy, confusion matrix, ROC-AUC, and precision-recall curves.
- **Remaining Work:** Development of Phase II decision-making model.
- **Lessons Learned:** Insights into project organization, time management, and the importance of structured version control.

**Detailed Phases:**
1. **Phase I – Remaining Useful Life (RUL) Prediction:** Predicting if a robot will remain operational over the next six months based on time series failure data.
   - **Evaluation Criteria:** Success metrics calculated based on actual vs. predicted RUL, with rewards and penalties for prediction accuracy.
2. **Phase II – Maintenance Decision-Making:** Assessing whether to replace robots before a new mission, with the goal of maintaining at least 10 operational robots.
   - **Evaluation Criteria:** Rewards for successful missions and penalties for robot failures or unnecessary replacements.

**Data Structure:**
- **`failure_data.csv`**: Summarizes failure modes and associated timestamps.
- **`Degradation data`**: Time series data tracking crack length evolution in each robot.
- **Testing Data**: Structured datasets for model validation based on real-world test scenarios.

**Implementation:**
The project is built using **Streamlit** for visualization. Key files include:
- **app.py**: Main application entry point.
- **dashboard/**: Contains dashboard components and layout.
- **data/**: Contains processed data and input files.

**Lessons Learned:**  
Insights gained include key considerations for model selection, the value of setting time-focused project goals, regular version control, and structuring the repository for collaboration.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Clone the Repository
```bash
git clone https://github.com/mriusero/predicitve-maintenance-on-industrial-robots
cd predicitve-maintenance-on-industrial-robots
```

## Create and Activate a Virtual Environment (Optional but Recommended)

### On macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### On Windows
```bash
python -m venv venv
venv\Scripts\activate
```

## Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Application
To start the Streamlit application, run the following command from the project’s root directory:

```bash
cd app 
streamlit run app/app.py
```

### Directory Structure
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

18 directories, 326 files
```

---

### **2. Training Data**
- **Source URL:** [training_data].
- **Number of Samples:** 50 total samples.
- **`failure_data.csv`:**
  - Summarizes times-to-failure for the 50 samples.
  - Indicates failure mode for each sample (Infant Mortality, Fatigue Crack, or Control Board Failure).
- **`degradation_data` folder:**
  - Contains crack length measurements for each sample.
  - Each CSV file in this folder corresponds to a specific sample, with `item_X` matching `item_id` in `failure_data.csv`.
- **`create_a_testing_dataset.ipynb` Notebook:**
  - Generates a “pseudo testing dataset” based on training data.
  - Demonstrates data generation and can be used to develop models using pseudo-test data.

### **3. Testing Data and Evaluation**
- **Test Data Link: `testing_data/sent_to_student/group_0`** 
- **`Sample_submission.csv`:**  
  A template for submitting results.
