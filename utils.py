import torch
from torch.utils.data import Dataset

class PianoDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Separate x and y
        x_notes = torch.tensor([pair[0] for pair in sequence], dtype=torch.long)  # Note indices
        x_durations = torch.tensor([pair[1] for pair in sequence], dtype=torch.float)  # Durations
        
        # Separate note and duration in the label
        y_note = torch.tensor(label[0], dtype=torch.long)  # Next note
        y_duration = torch.tensor(label[1], dtype=torch.float)  # Next duration
        
        return x_notes, x_durations, y_note, y_duration
