import streamlit as st

#from src.forecasting.main import handle_fleet_management


def page_5():
    st.markdown('<div class="header">#5 Fleet Management_</div>', unsafe_allow_html=True)
    st.markdown("""
    Here is the phase that consist to use conjointly the two models to orchest the fleet management of industrial robots.
    
    ---
    """)

    st.write("""
    
    ## Organisation

    ### 1. **Train, Validate and Test splitting**:

    The purpose of this phase is to define the strategy to adapt models to the data distribution probabilistically evolving.

    * **Train set** with the `train_df` *(contains the data for training the model with true RUL values and failures modes)*
    * **Validation set** with the `pseudo_test_with_truth_df` *(contains the data for validating the model with true RUL values but no failures modes)*
    * **Test set** with the `test_df` *(contains the data for testing the model with no true RUL values and no failures modes)*
        
    > *Validation & test sets are probabilistically generated and distributions are evolving constantly at each data generation with the button `[Update data]`.*
    
    
    ### 2. **Fleet Management**:
        
    The purpose of this phase is to use the two models to orchestrate the fleet management of industrial robots according to the Phase II of the Kaggle competition.
    
    #### 2.1 - **Survival prediction** with 'rul_survival_predictor'
        
    ```python
    Determination if a model have less or more than 50% of chance to survive in the next 6 months.  # RUL binary classification
    ```
                
    #### 2.2 - **Crack growth forecast** with 'lstm_based_crack_forecaster'

    ```python
    if RUL > 6 months : # (results from RUL binary classification)
    
        --> determination of the crack length evolution for the next 6 months.
        
    else: # RUL < 6 months
    
        --> classification of the failure mode of the robot. # Within 'Infant mortality', 'Control board failure' or 'Fatigue crack'.
    ```
                    
    #### 2.3 - **Fleet Management**:
    ```python
    
    --> For each scenario, at a (t) instant (start of the 6 months mission):
    - Analyse the current state of the robot fleet (RUL < 6 months or not) and choose 10 robots within a batch of 12 robots.
    - Decide the necessary replacements based on the risks for the next 6 months. (Strategy = 'Replace' or 'No action Required')
    
    --> Repeat the process for 20 consecutive missions, covering 120 months. 
    
    > Some robots the time does not start from zero. This is because it has failed and was replaced before. In the csv file, the time starts from the previous replacement of the robot. It is assumed that after replacement, the robot is as-good-as-new.
    
    ```
    
    
    
    
    """)


   # handle_fleet_management()
