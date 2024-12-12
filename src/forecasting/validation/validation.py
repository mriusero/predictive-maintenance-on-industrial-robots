# validation.py
import pandas as pd
import os

def generate_submission_file(model_name, submission_path, step):

    template = pd.read_csv('submission/template/submission_template_phase_1.csv')
    submission_df = template.copy()
    submission_df['label'] = 0


    if model_name == 'lstm_based_crack_forecaster':

        lstm_results = pd.read_csv(f"{submission_path}/lstm_predictions_{step}.csv")
        lstm_results = lstm_results[['item_id',
                                   #  'Infant mortality', 'Control board failure', 'Fatigue crack',
                                     'length_measured', 'length_filtered']]

        for item_index, group in lstm_results.groupby('item_id'):
            formatted_item_index = f"item_{item_index}"
            if (group['length_measured'] > 0.85).any():
                submission_df.loc[submission_df['item_index'] == formatted_item_index, 'label'] = 1


    if model_name == 'rul_survival_predictor':

        gbsa_results = pd.read_csv(f"{submission_path}/{model_name}_{step}.csv")
        gbsa_results['item_id'] = gbsa_results['item_id'].astype(int)
        for item_index, group in gbsa_results.groupby('item_id'):
            formatted_item_index = f"item_{item_index}"
            #print(formatted_item_index)
            if (group['predicted_failure_6_months_binary'] == 1).any():
                submission_df.loc[submission_df['item_index'] == formatted_item_index, 'label'] = int(1)
    else:
        raise ValueError("'model_name' not defined in 'generate_submission_file()'")

    submission_path = os.path.abspath(submission_path)
    print('Submission file available at:', f"file://{submission_path}/submission_{step}.csv")
    return submission_df.sort_values('item_index').to_csv(f"{submission_path}/submission_{step}.csv", index=False)
