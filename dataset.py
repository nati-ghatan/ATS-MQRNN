import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import numpy as np



class MQRNN_dataset(Dataset):
    def __init__(self,
                 target_dataframe,
                 horizon_size,
                 covariate_size,
                 batch_size,
                 train_frac=0.9,
                 std_epsilon=0.01):
        self.std_epsilon = std_epsilon
        self.target_dataframe = target_dataframe
        self.horizon_size = horizon_size
        self.covariate_size = covariate_size
        self.batch_size = batch_size

        assert 0 <= train_frac <= 1
        # Normalize data-frame using Z-score
        self.target_dataframe = ((self.target_dataframe / self.target_dataframe[self.target_dataframe != 0].mean()) - 1)
        # self.target_dataframe = ((self.target_dataframe - self.target_dataframe[self.target_dataframe != 0].mean()) \
        #                         / (self.target_dataframe[self.target_dataframe != 0].std() + self.std_epsilon)) * 100

        self.covariates_df = self._create_covariates_df()
        self.next_covariate = self._create_full_covariates_df()

        # Split to train and test sets
        n_series = self.target_dataframe.shape[1]
        train_size = int((n_series * train_frac) - (n_series * train_frac) % self.batch_size)
        self.train_indices = sorted(np.random.choice(n_series, train_size, replace=False))
        self.test_indices = sorted(list(set(range(n_series)) - set(self.train_indices)))

        self.train_target_df = self.target_dataframe.iloc[:, self.train_indices]
        self.test_target_df = self.target_dataframe.iloc[:, self.test_indices]

    def _create_covariates_df(self):
        yearly = np.sin(2 * np.pi * self.target_dataframe.index.dayofyear / 366)
        weekly = np.sin(2 * np.pi * self.target_dataframe.index.dayofweek / 7)
        daily = np.sin(2 * np.pi * self.target_dataframe.index.hour / 24)
        return pd.DataFrame({'yearly': yearly,
                             'weekly': weekly,
                             'daily': daily},
                            index=self.target_dataframe.index)

    def _create_full_covariates_df(self):
        full_covariate = []
        for i in range(1, self.target_dataframe.shape[0] - self.horizon_size + 1):
            new_entry = self.covariates_df.iloc[i:i + self.horizon_size, :].to_numpy()
            full_covariate.append(new_entry)

        full_covariate = np.array(full_covariate)
        return full_covariate.reshape(-1, self.horizon_size * self.covariate_size)

    def __get_item(self, idx, target_df):
        cur_series = np.array(target_df.iloc[: -self.horizon_size, idx])
        cur_covariate = np.array(
            self.covariates_df.iloc[:-self.horizon_size, :])  # covariate used in generating hidden states

        real_vals_list = []
        for i in range(1, self.horizon_size + 1):
            real_vals_list.append(
                np.array(target_df.iloc[i: target_df.shape[0] - self.horizon_size + i, idx]))
        real_vals_array = np.array(real_vals_list)  # [horizon_size, seq_len]
        real_vals_array = real_vals_array.T  # [seq_len, horizon_size]
        cur_series_tensor = torch.tensor(cur_series)

        cur_series_tensor = torch.unsqueeze(cur_series_tensor, dim=1)  # [seq_len, 1]
        cur_covariate_tensor = torch.tensor(cur_covariate)  # [seq_len, covariate_size]
        cur_series_covariate_tensor = torch.cat([cur_series_tensor, cur_covariate_tensor], dim=1)
        next_covariate_tensor = torch.tensor(self.next_covariate)  # [seq_len, horizon_size * covariate_size]

        cur_real_vals_tensor = torch.tensor(real_vals_array)
        return cur_series_covariate_tensor, next_covariate_tensor, cur_real_vals_tensor

    def test_length(self):
        return self.test_target_df.shape[1]

    def get_test_item(self, idx):
        return self.__get_item(idx=idx, target_df=self.test_target_df)

    def __len__(self):
        return self.train_target_df.shape[1]

    def __getitem__(self, idx):
        return self.__get_item(idx=idx, target_df=self.train_target_df)
