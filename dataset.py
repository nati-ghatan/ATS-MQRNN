import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class MQRNN_dataset(Dataset):
    def __init__(self, target_dataframe, horizon_size, covariate_size):
        self.target_dataframe = target_dataframe
        self.horizon_size = horizon_size
        self.covariate_size = covariate_size

        yearly = np.sin(2 * np.pi * target_dataframe.index.dayofyear / 366)
        weekly = np.sin(2 * np.pi * target_dataframe.index.dayofweek / 7)
        daily = np.sin(2 * np.pi * target_dataframe.index.hour / 24)
        self.covariates_df = pd.DataFrame({'yearly': yearly,
                                           'weekly': weekly,
                                           'daily': daily},
                                          index=target_dataframe.index)

        full_covariate = []
        for i in range(1, target_dataframe.shape[0] - horizon_size + 1):
            new_entry = self.covariates_df.iloc[i:i + horizon_size, :].to_numpy()
            full_covariate.append(new_entry)

        full_covariate = np.array(full_covariate)
        full_covariate = full_covariate.reshape(-1, horizon_size * covariate_size)

        self.next_covariate = full_covariate

    def __len__(self):
        return self.target_dataframe.shape[1]

    def __getitem__(self, idx):
        cur_series = np.array(self.target_dataframe.iloc[: -self.horizon_size, idx])
        cur_covariate = np.array(
            self.covariates_df.iloc[:-self.horizon_size, :])  # covariate used in generating hidden states

        real_vals_list = []
        for i in range(1, self.horizon_size + 1):
            real_vals_list.append(
                np.array(self.target_dataframe.iloc[i: self.target_dataframe.shape[0] - self.horizon_size + i, idx]))
        real_vals_array = np.array(real_vals_list)  # [horizon_size, seq_len]
        real_vals_array = real_vals_array.T  # [seq_len, horizon_size]
        cur_series_tensor = torch.tensor(cur_series)

        cur_series_tensor = torch.unsqueeze(cur_series_tensor, dim=1)  # [seq_len, 1]
        cur_covariate_tensor = torch.tensor(cur_covariate)  # [seq_len, covariate_size]
        cur_series_covariate_tensor = torch.cat([cur_series_tensor, cur_covariate_tensor], dim=1)
        next_covariate_tensor = torch.tensor(self.next_covariate)  # [seq_len, horizon_size * covariate_size]

        cur_real_vals_tensor = torch.tensor(real_vals_array)
        return cur_series_covariate_tensor, next_covariate_tensor, cur_real_vals_tensor
