# General imports
import numpy as np
import pandas as pd
import pprint

# Our implementation imports
from dataset import MQRNN_dataset
from MQRNN import MQRNN


# Debug methods
def __prepare_data():
    # Load parquet dataframe
    eldata = pd.read_parquet("LD2011_2014.parquet")
    n_rows_before_resampling = eldata.shape[0]

    # Resample from 15-min to 1-hr frequency
    eldata = eldata.resample("1H", on="timestamp").mean()  # Resample data on an hourly frequency
    n_rows_after_resampling = eldata.shape[0]

    # Verify that resampling was done correctly
    assert (n_rows_before_resampling / 4) == (n_rows_after_resampling - 1)

    return eldata


def __prepare_data_for_training(target_dataframe, horizon_size, covariate_size):
    yearly = np.sin(2 * np.pi * target_dataframe.index.dayofyear / 366)
    weekly = np.sin(2 * np.pi * target_dataframe.index.dayofweek / 7)
    daily = np.sin(2 * np.pi * target_dataframe.index.hour / 24)
    covariates_df = pd.DataFrame({'yearly': yearly, 'weekly': weekly, 'daily': daily}, index=target_dataframe.index)

    full_covariate = []
    for i in range(1, target_dataframe.shape[0] - horizon_size + 1):
        new_entry = covariates_df.iloc[i:i + horizon_size, :].to_numpy()
        full_covariate.append(new_entry)

    full_covariate = np.array(full_covariate)
    full_covariate = full_covariate.reshape(-1, horizon_size * covariate_size)


def main():
    # Running parameters
    batch_size = 1  # 128
    hidden_size = 8
    covariate_size = 3
    horizon_size = 24
    context_size = 10
    quantiles = [0.25, 0.5, 0.75]
    device = 'cpu'
    learning_rate = 1e-3
    num_epochs = 2

    # Load and preprocess data
    eldata = __prepare_data()  # TODO: Decide if we want/need to scale our data like Gleb did
    eldata = eldata[-1000:]  # TODO: Remove this after implementation is complete

    # Define dataset
    # TODO: Need to add division into train vs. test sets (we can skip validation)
    dataset = MQRNN_dataset(target_dataframe=eldata,
                            horizon_size=horizon_size,
                            covariate_size=covariate_size)

    # Verify dataset was created correctly
    # Comments show expected shapes FOR ALL FCTs:
    cur_series_covariate_tensor, next_covariate_tensor, cur_real_vals_tensor = dataset[0]
    print(f"\nInput shapes:\n#############")
    print(f"cur_series_covariate_tensor:    {cur_series_covariate_tensor.shape}")  # Target + covariates
    print(f"next_covariate_tensor:          {next_covariate_tensor.shape}")  # Future covariates (flattened)
    print(f"cur_real_vals_tensor:           {cur_real_vals_tensor.shape}")  # Real expected target values

    # Define model
    model = MQRNN(horizon_size, hidden_size, quantiles,
                  learning_rate, batch_size, num_epochs,
                  context_size, covariate_size, device)

    # Test forward method
    # print(f"\nForward results:\n################")
    # forecasts = model.forward(cur_series_covariate_tensor, next_covariate_tensor)
    # print(f"Forecasts shape:                {forecasts.shape}")

    # Train
    print("Training model, stand by...\n###########################")
    model.train(dataset=dataset, n_epochs_per_report=1)

    # Test predict method
    print(f"\nPrediction results:\n###################")
    # Reshape input for a single FCT
    cur_series_covariate_tensor = cur_series_covariate_tensor[0, :].reshape(1, covariate_size + 1)
    next_covariate_tensor = next_covariate_tensor[0, :].reshape(1, covariate_size * horizon_size)

    # Make prediction
    prediction = model.predict(cur_series_covariate_tensor, next_covariate_tensor)
    pprint.pprint(prediction, width=1)


if __name__ == "__main__":
    main()
