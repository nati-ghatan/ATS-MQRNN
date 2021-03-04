import torch
from models import Encoder, GlobalDecoder, LocalDecoder
from temp_data import MQRNN_dataset

# TODO: Create our own version and change these imports
from temp_train_func import train_fn


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

    def forward(self, cur_series_covariate_tensor, next_covariate_tensor):
        # Expected argument shapes:
        # cur_series_covariate_tensor - [seq_len, target_size + covariate_size]
        # next_covariate_tensor -       [seq_len, covariate_size * horizon_size]

        # Reshape input data - [seq_len, batch_size, target_size + covariate_size]
        input_data = cur_series_covariate_tensor.reshape(cur_series_covariate_tensor.shape[0],
                                                         self.batch_size,
                                                         cur_series_covariate_tensor.shape[1])
        # Encode and reshape hidden state - [seq_len, batch_size, hidden_size]
        hidden_state = self.encoder(input_data)
        hidden_state = hidden_state.permute(1, 0, 2)

        # Reshape next covariate tensor - [seq_len, covariate_size * horizon_size]
        next_covariate_tensor = next_covariate_tensor.reshape(next_covariate_tensor.shape[0],
                                                              self.batch_size,
                                                              next_covariate_tensor.shape[1])

        # Decode horizon-specific and -agnostic contexts
        # Expected input shape - [seq_len, batch_size, hidden_size+covariate_size * horizon_size]
        # Expected contexts shape - [seq_len, batch_size, (horizon_size+1) * context_size]
        hidden_and_covariate = torch.cat([hidden_state, next_covariate_tensor], dim=2)
        contexts = self.gdecoder(hidden_and_covariate)

        # Forecast quantiles over all horizons
        # Expected input shape - [seq_len, batch_size,(horizon_size+1)*context_size + covariate_size * horizon_size]
        # Expected forecasts shape - [seq_len, batch_size, horizon_size* quantile_size]
        local_decoder_input = torch.cat([contexts, next_covariate_tensor], dim=2)
        forecasts = self.ldecoder(local_decoder_input)

        return forecasts

    def train(self, dataset: MQRNN_dataset):
        # TODO: Implement this on our own
        train_fn(encoder=self.encoder,
                 gdecoder=self.gdecoder,
                 ldecoder=self.ldecoder,
                 dataset=dataset,
                 lr=self.lr,
                 batch_size=self.batch_size,
                 num_epochs=self.num_epochs,
                 device=self.device)
        print("training finished")

    def predict(self, train_target_df, train_covariate_df, test_covariate_df, col_name):
        # TODO: Implement this on our own
        input_target_tensor = torch.tensor(train_target_df[[col_name]].to_numpy())
        full_covariate = train_covariate_df.to_numpy()
        full_covariate_tensor = torch.tensor(full_covariate)

        next_covariate = test_covariate_df.to_numpy()
        next_covariate = next_covariate.reshape(-1, self.horizon_size * self.covariate_size)
        next_covariate_tensor = torch.tensor(next_covariate)  # [1,horizon_size * covariate_size]

        input_target_tensor = input_target_tensor.to(self.device)
        full_covariate_tensor = full_covariate_tensor.to(self.device)
        next_covariate_tensor = next_covariate_tensor.to(self.device)

        with torch.no_grad():
            input_target_covariate_tensor = torch.cat([input_target_tensor, full_covariate_tensor], dim=1)
            input_target_covariate_tensor = torch.unsqueeze(input_target_covariate_tensor,
                                                            dim=0)  # [1, seq_len, 1+covariate_size]
            input_target_covariate_tensor = input_target_covariate_tensor.permute(1, 0,
                                                                                  2)  # [seq_len, 1, 1+covariate_size]
            print(f"input_target_covariate_tensor shape: {input_target_covariate_tensor.shape}")
            outputs = self.encoder(input_target_covariate_tensor)  # [seq_len,1,hidden_size]
            hidden = torch.unsqueeze(outputs[-1], dim=0)  # [1,1,hidden_size]

            next_covariate_tensor = torch.unsqueeze(next_covariate_tensor, dim=0)
            # next_covariate_tensor = torch.unsqueeze(next_covariate_tensor, dim=0) # [1,1, covariate_size * horizon_size]

            print(f"hidden shape: {hidden.shape}")
            print(f"next_covariate_tensor: {next_covariate_tensor.shape}")
            gdecoder_input = torch.cat([hidden, next_covariate_tensor],
                                       dim=2)  # [1,1, hidden + covariate_size* horizon_size]
            gdecoder_output = self.gdecoder(gdecoder_input)  # [1,1,(horizon_size+1)*context_size]

            local_decoder_input = torch.cat([gdecoder_output, next_covariate_tensor],
                                            dim=2)  # [1, 1,(horizon_size+1)*context_size + covariate_size * horizon_size]
            local_decoder_output = self.ldecoder(
                local_decoder_input)  # [seq_len, batch_size, horizon_size* quantile_size]
            local_decoder_output = local_decoder_output.view(self.horizon_size, self.quantile_size)
            output_array = local_decoder_output.cpu().numpy()
            result_dict = {}
            for i in range(self.quantile_size):
                result_dict[self.quantiles[i]] = output_array[:, i]
            return result_dict
