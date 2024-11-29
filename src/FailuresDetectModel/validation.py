# validation.py
import pandas as pd

def generate_submission_file(model_name, output_path, step):

    template = pd.read_csv('../app/data/output/submission/template/submission_template.csv')
    submission_df = template.copy()
    submission_df['label'] = 0
    submission_df['predicted_rul'] = 0

    if model_name == 'LSTMModel':

        lstm_results = pd.read_csv(f"{output_path}/lstm_predictions_{step}.csv")
        lstm_results = lstm_results[['item_id',
                                   #  'Infant mortality', 'Control board failure', 'Fatigue crack',
                                     'length_measured', 'length_filtered']]

        for item_index, group in lstm_results.groupby('item_id'):
            formatted_item_index = f"item_{item_index}"
            if (group['length_measured'] > 0.85).any():
                submission_df.loc[submission_df['item_index'] == formatted_item_index, 'label'] = 1

            #if (group['crack_failure'] == 1).any():
            #    submission_df.loc[submission_df['item_index'] == formatted_item_index, 'label'] = 1

            #control_board_failure = row['control_board_failure_filtered']
            #if control_board_failure == 1:
            #    submission_df.loc[submission_df['item_index'] == item_index, 'label'] = int(1)

        #for index, row in lstm_results.iterrows():
        #    item_index = row['item_index']
        #    failure = row['Failure mode (lstm)']
        #    #if failure == 'Crack failure':
        #        #submission_df.loc[submission_df['item_index'] == item_index, 'label'] = 1
        #    if failure == 'Control board failure':
        #        submission_df.loc[submission_df['item_index'] == item_index, 'label'] = int(1)
        #    if failure == 'Infant Mortality':
        #        submission_df.loc[submission_df['item_index'] == item_index, 'label'] = int(1)

    if model_name == 'RandomForestClassifierModel':

        rf_results = pd.read_csv(f"{output_path}/rf_predictions_{step}.csv")

        for item_index, group in rf_results.groupby('item_id'):
            formatted_item_index = f"item_{item_index}"

            if (group['Failure mode (rf)'] == 'Control board failure').any():
                submission_df.loc[submission_df['item_index'] == formatted_item_index, 'label'] = int(1)
            elif (group['Failure mode (rf)'] == 'Infant mortality').any():
                submission_df.loc[submission_df['item_index'] == formatted_item_index, 'label'] = int(1)

    if model_name == 'GradientBoostingSurvivalModel':

        gbsa_results = pd.read_csv(f"{output_path}/gbsa_predictions_{step}.csv")
        gbsa_results['item_id'] = gbsa_results['item_id'].astype(int)
        for item_index, group in gbsa_results.groupby('item_id'):
            formatted_item_index = f"item_{item_index}"
            #print(formatted_item_index)
            if (group['predicted_failure_6_months_binary'] == 1).any():
                submission_df.loc[submission_df['item_index'] == formatted_item_index, 'label'] = int(1)
    else:
        raise ValueError("'model_name' not defined in 'generate_submission_file()'")

    return submission_df.sort_values('item_index').to_csv(f"{output_path}/submission_{step}.csv", index=False)

def calculate_score(output_path, step):

    solution = pd.read_csv('../app/data/input/training_data/pseudo_testing_data_with_truth/Solution.csv')
    submission = pd.read_csv(f"{output_path}/submission_{step}.csv")

    reward = 2
    penalty_false_positive = -1 / 60
    penalty_false_negative = -4

    # Compare labels and calculate rewards/penalties
    rewards_penalties = []
    for _, (sol_label, sub_label, true_rul) in enumerate(
            zip(solution['label'], submission['label'], solution['true_rul'])):
        if sol_label == sub_label:
            rewards_penalties.append(reward)
        elif sol_label == 1 and sub_label == 0:
            rewards_penalties.append(penalty_false_negative)
        elif sol_label == 0 and sub_label == 1:
            rewards_penalties.append(penalty_false_positive * true_rul)
        else:
            rewards_penalties.append(0)  # No reward or penalty if labels don't match

    return sum(rewards_penalties)