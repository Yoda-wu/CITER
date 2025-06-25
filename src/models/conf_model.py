import torch
from torchvision.ops import MLP
import torch.nn as nn


class ConfidenceModel(nn.Module):
    def __init__(self, config):
        super(ConfidenceModel, self).__init__()
        self.config = config
        self.mlp = MLP(
            in_channels=config.hidden_size,
            hidden_channels=[2 * config.hidden_size, config.hidden_size],
            norm_layer=nn.BatchNorm1d,
            activation_layer=nn.ReLU,
            dropout=0.1,
        )
        self.linear = nn.Linear(config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        if self.config.pad_token_id is None:
            self.config.pad_token_id = self.config.eos_token_id

    def forward(self, hidden_states, token_indices=None) -> torch.Tensor:

        # reshape to batch_size * seq_length, hidden_size
        batch_size, seq_length, hidden_size = hidden_states.size()
        mlp_input = hidden_states.view(-1, hidden_size)

        # pass all embeddings to mlp
        mlp_output = self.mlp(mlp_input)
        linear_output = self.linear(mlp_output)

        # reshape to batch_size, seq_length, 1
        scores = linear_output.view(batch_size, seq_length, -1)

        if token_indices is not None:
            # extract the score for the specified token indices
            confidence_scores = torch.gather(
                scores, 1, token_indices.unsqueeze(-1)
            ).squeeze(-1)
        else:
            confidence_scores = scores

        return self.sigmoid(confidence_scores)
