import pytorch_lightning as pl
from torch import nn


class Encoder(pl.LightningModule):
    """Encoder network for encoder-decoder forecast model."""

    def __init__(self, hist_len=168, fct_len=24, input_size=4, num_layers=1, hidden_units=8, lr=1e-3):
        super().__init__()

        self.hist_len = hist_len
        self.fct_len = fct_len
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_units = hidden_units
        self.lr = lr

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_units,
                            num_layers=self.num_layers,
                            batch_first=True)

    def forward(self, x, hidden=None):
        output, (hh, cc) = self.lstm(x)
        return hh


class GlobalDecoder(nn.Module):
    """
    Global decoder that receives the encoder's hidden state
    plus the covariates and returns both horizon-specific &
    -agnostic contexts (to be used in the local decoder)
    input_size = hidden_size + covariate_size * horizon_size
    output_size: (horizon_size+1) * context_size
    """

    def __init__(self,
                 hidden_size: int,
                 covariate_size: int,
                 horizon_size: int,
                 context_size: int):
        super(GlobalDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.covariate_size = covariate_size
        self.horizon_size = horizon_size
        self.context_size = context_size

        self.linear1 = nn.Linear(in_features=hidden_size + covariate_size * horizon_size,
                                 out_features=horizon_size * hidden_size * 3)

        self.linear2 = nn.Linear(in_features=horizon_size * hidden_size * 3,
                                 out_features=horizon_size * hidden_size * 2)

        self.linear3 = nn.Linear(in_features=horizon_size * hidden_size * 2,
                                 out_features=(horizon_size + 1) * context_size)

        self.activation = nn.ReLU()

    def forward(self, input):
        layer1_output = self.linear1(input)
        layer1_output = self.activation(layer1_output)

        layer2_output = self.linear2(layer1_output)
        layer2_output = self.activation(layer2_output)

        layer3_output = self.linear3(layer2_output)
        layer3_output = self.activation(layer3_output)
        return layer3_output


class LocalDecoder(nn.Module):
    """
    Local decoder that receives the Global decoder's contexts
    (both horizon-specific & -agnostic) plus future covariates
    and forecasts the required quantiles over all horizons
    input_size: (horizon_size+1)*context_size + horizon_size*covariate_size
    output_size: horizon_size * quantile_size
    """

    def __init__(self,
                 covariate_size,
                 quantile_size,
                 context_size,
                 quantiles,
                 horizon_size):
        super(LocalDecoder, self).__init__()
        self.covariate_size = covariate_size
        self.quantiles = quantiles
        self.quantile_size = quantile_size
        self.horizon_size = horizon_size
        self.context_size = context_size

        self.linear1 = nn.Linear(in_features=horizon_size * context_size + horizon_size * covariate_size + context_size,
                                 out_features=horizon_size * context_size)
        self.linear2 = nn.Linear(in_features=horizon_size * context_size,
                                 out_features=horizon_size * quantile_size)
        self.activation = nn.ReLU()

    def forward(self, input):
        layer1_output = self.linear1(input)
        layer1_output = self.activation(layer1_output)

        layer2_output = self.linear2(layer1_output)
        layer2_output = self.activation(layer2_output)
        return layer2_output
