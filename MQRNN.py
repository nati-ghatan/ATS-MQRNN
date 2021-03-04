import torch
from models import Encoder, GlobalDecoder, LocalDecoder
from dataset import MQRNN_dataset
from torch.utils.data import DataLoader


class MQRNN(object):
    """
    This class holds the encoder and the global decoder and local decoder.
    """

    def __init__(self,
                 horizon_size: int,
                 hidden_size: int,
                 quantiles: list,
                 lr: float,
                 batch_size: int,
                 num_epochs: int,
                 context_size: int,
                 covariate_size: int,
                 device):
        # Notify user of selected device
        print(f"Selected device is: {device}")

        # Initialize model parameters
        self.device = device
        self.horizon_size = horizon_size
        self.hidden_size = hidden_size
        self.quantiles = quantiles
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.context_size = context_size
        self.covariate_size = covariate_size
        self.quantile_size = len(quantiles)

        # Define encoder-decoder sub-components
        self.__initialize_sub_components()

    # Private methods

    def __initialize_sub_components(self):
        # Define encoder & decoder sub-components
        self.encoder = Encoder(lr=self.lr,
                               hidden_units=self.hidden_size,
                               num_layers=1)

        self.gdecoder = GlobalDecoder(hidden_size=self.hidden_size,
                                      covariate_size=self.covariate_size,
                                      horizon_size=self.horizon_size,
                                      context_size=self.context_size)
        self.ldecoder = LocalDecoder(covariate_size=self.covariate_size,
                                     quantile_size=self.quantile_size,
                                     context_size=self.context_size,
                                     quantiles=self.quantiles,
                                     horizon_size=self.horizon_size)
        # Apply double casting
        self.encoder.double()
        self.gdecoder.double()
        self.ldecoder.double()

    def __reshape_tensor_for_decoder(self, input_tensor: torch.Tensor, n_samples, dim_without_batch=2):
        # Final output shape should be: [seq_len, batch_size, input_tensor_contents]
        if len(input_tensor.shape) == dim_without_batch:
            return input_tensor.reshape(input_tensor.shape[0], n_samples, input_tensor.shape[1])
        else:
            return input_tensor.permute(1, 0, 2)

    def __compute_loss(self, cur_series_covariate_tensor: torch.Tensor,
                       next_covariate_tensor: torch.Tensor,
                       cur_real_vals_tensor: torch.Tensor):

        # Initialize variables
        total_loss = torch.tensor([0.0], device=self.device)

        # Cast to double
        cur_series_covariate_tensor = cur_series_covariate_tensor.double()  # [batch_size, seq_len, 1+covariate_size]
        next_covariate_tensor = next_covariate_tensor.double()  # [batch_size, seq_len, covariate_size * horizon_size]
        cur_real_vals_tensor = cur_real_vals_tensor.double()  # [batch_size, seq_len, horizon_size]

        # Switch devices
        cur_series_covariate_tensor = cur_series_covariate_tensor.to(self.device)
        next_covariate_tensor = next_covariate_tensor.to(self.device)
        cur_real_vals_tensor = cur_real_vals_tensor.to(self.device)
        self.encoder.to(self.device)
        self.gdecoder.to(self.device)
        self.ldecoder.to(self.device)

        # Run forward
        forecasts = self.forward(cur_series_covariate_tensor=cur_series_covariate_tensor,
                                 next_covariate_tensor=next_covariate_tensor,
                                 n_samples=self.batch_size)

        # Reshape real values tensor - [seq_len, batch_size, horizon_size]
        cur_real_vals_tensor = self.__reshape_tensor_for_decoder(input_tensor=cur_real_vals_tensor,
                                                                 n_samples=self.batch_size)

        # Compute total quantile loss
        for i in range(self.quantile_size):
            p = self.ldecoder.quantiles[i]
            errors = cur_real_vals_tensor - forecasts[:, :, :, i]
            cur_loss = torch.max((p - 1) * errors, p * errors)
            total_loss += torch.sum(cur_loss)

        return total_loss

    # Public methods
    def forward(self, cur_series_covariate_tensor: torch.Tensor, next_covariate_tensor: torch.Tensor, n_samples: int):
        # Expected argument shapes:
        # cur_series_covariate_tensor - [seq_len, target_size + covariate_size]
        # next_covariate_tensor -       [seq_len, covariate_size * horizon_size]

        # Reshape input data - [seq_len, batch_size, target_size + covariate_size]
        encoder_input = self.__reshape_tensor_for_decoder(input_tensor=cur_series_covariate_tensor, n_samples=n_samples)

        # Encode and reshape hidden state - [seq_len, batch_size, hidden_size]
        hidden_state = self.encoder(encoder_input)
        hidden_state = hidden_state.permute(1, 0, 2)
        hidden_state = hidden_state.repeat(1, n_samples, 1)

        # Reshape next covariate tensor - [seq_len, covariate_size * horizon_size]
        decoder_covariate_input = self.__reshape_tensor_for_decoder(input_tensor=next_covariate_tensor, n_samples=n_samples)

        # Decode horizon-specific and -agnostic contexts
        # Expected input shape - [seq_len, batch_size, hidden_size+covariate_size * horizon_size]
        # Expected contexts shape - [seq_len, batch_size, (horizon_size+1) * context_size]
        # print('-'*20)
        # print('hidden_state', hidden_state.shape)
        # print('decoder_covariate_input', decoder_covariate_input.shape)
        hidden_and_covariate = torch.cat([hidden_state, decoder_covariate_input], dim=2)
        contexts = self.gdecoder(hidden_and_covariate)

        # Forecast quantiles over all horizons
        # Expected input shape - [seq_len, batch_size,(horizon_size+1)*context_size + covariate_size * horizon_size]
        # Expected forecasts shape - [seq_len, batch_size, horizon_size* quantile_size]
        local_decoder_input = torch.cat([contexts, decoder_covariate_input], dim=2)
        forecasts = self.ldecoder(local_decoder_input)

        return forecasts.reshape(self.seq_len, n_samples, self.horizon_size, self.quantile_size)

    def train(self, dataset: MQRNN_dataset, n_epochs_per_report=-1):
        # Initialize optimizers
        encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        gdecoder_optimizer = torch.optim.Adam(self.gdecoder.parameters(), lr=self.lr)
        ldecoder_optimizer = torch.optim.Adam(self.ldecoder.parameters(), lr=self.lr)

        data_iter = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        loss_per_epoch = []
        for epoch_index in range(self.num_epochs):
            epoch_loss_sum = 0.0
            total_sample = 0
            for i, (cur_series_tensor, cur_covariate_tensor, cur_real_vals_tensor) in enumerate(data_iter):
                # if i in (165,166):
                #     print('cur_series_tensor', cur_series_tensor.shape)
                #     print('cur_covariate_tensor', cur_covariate_tensor.shape)
                #     print('cur_real_vals_tensor', cur_real_vals_tensor.shape)
                self.seq_len = cur_series_tensor.shape[1]
                horizon_size = cur_covariate_tensor.shape[-1]
                total_sample += self.batch_size * self.seq_len * horizon_size
                encoder_optimizer.zero_grad()
                gdecoder_optimizer.zero_grad()
                ldecoder_optimizer.zero_grad()
                loss = self.__compute_loss(cur_series_tensor, cur_covariate_tensor, cur_real_vals_tensor)
                loss.backward()
                encoder_optimizer.step()
                gdecoder_optimizer.step()
                ldecoder_optimizer.step()
                epoch_loss_sum += loss.item()
            epoch_loss_mean = epoch_loss_sum / total_sample
            loss_per_epoch.append(epoch_loss_mean)
            if (n_epochs_per_report >= 1) and ((epoch_index + 1) % n_epochs_per_report == 0):
                print(f"\tEpoch {epoch_index + 1} of {self.num_epochs}, loss = {epoch_loss_mean}")
        print("Training completed successfully!")
        return loss_per_epoch

    def predict(self, cur_series_covariate_tensor: torch.Tensor, next_covariate_tensor: torch.Tensor):
        self.seq_len = 1

        # Make forecasts (no need to track gradients in prediction time)
        with torch.no_grad():
            forecasts = self.forward(cur_series_covariate_tensor=cur_series_covariate_tensor,
                                     next_covariate_tensor=next_covariate_tensor,
                                     n_samples=1)

        # Organize results in a readable format
        horizons = {}
        for horizon_index in range(self.horizon_size):
            quantile_predictions = {}
            for quantile_index in range(self.quantile_size):
                quantile_name = self.quantiles[quantile_index]
                quantile_predictions[quantile_name] = float(forecasts[:, :, horizon_index, quantile_index])
            horizons[f"t+{horizon_index + 1}"] = quantile_predictions
        return horizons