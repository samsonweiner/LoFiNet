import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim1, hidden_dim2):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim + 1, hidden_dim1, batch_first=True)  # Includes duration as input
        self.fc_hidden = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc_note = nn.Linear(hidden_dim2, vocab_size)  # Predict next note
        self.fc_duration = nn.Linear(hidden_dim2, 1)  # Predict next duration

    def forward(self, x_notes, x_durations):
        # Embedding for notes/chords
        x_embed = self.embedding(x_notes)
        
        # Concatenate with duration
        x_combined = torch.cat((x_embed, x_durations.unsqueeze(-1)), dim=-1)
        
        # LSTM
        lstm_out, _ = self.lstm(x_combined)
        
        # Output heads
        hidden_out = F.relu(self.fc_hidden(lstm_out[:, -1, :]))
        note_pred = self.fc_note(hidden_out)  # Last time step
        duration_pred = self.fc_duration(hidden_out)
        
        return note_pred, duration_pred