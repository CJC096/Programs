import torch
import torch.nn as nn


class RNNModule(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0):
        super(RNNModule, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)  # Add dropout here
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Convert input to LongTensor
        x = x.long()  # Convert to LongTensor

        # Embedding layer
        embedded = self.embedding(x)

        # RNN layer
        out, _ = self.rnn(embedded, h0)

        # Only take the output from the final time step
        out = self.fc(out[:, -1, :])
        return out

    def get_rnn_params(self):
        return {
            "rnn.weight_ih_l0": self.rnn.weight_ih_l0,
            "rnn.weight_hh_l0": self.rnn.weight_hh_l0,
            "rnn.bias_ih_l0": self.rnn.bias_ih_l0,
            "rnn.bias_hh_l0": self.rnn.bias_hh_l0
        }