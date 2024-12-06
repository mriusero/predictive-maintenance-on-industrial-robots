


st.markdown("# Below are previous test for phase I:\n______________________________________")
# --------- LSTM Model ------------
lstm_model = instance_model('LSTMModel')
lstm_predictions_cross_val, lstm_predictions_final_test = lstm_model.run_full_pipeline(train_df,
                                                                                       pseudo_test_with_truth_df,
                                                                                       test_df)

st.dataframe(lstm_predictions_cross_val)
st.dataframe(lstm_predictions_final_test)