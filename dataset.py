import pandas as pd
import torch
from torch.utils.data import Dataset

class VoiceDataset(Dataset):
    """class dataset with pytorch to create the train and test datasets"""

    def __init__(self, csv_file):
        #open the file and read it
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        #return the length of the dataset
        return len(self.df)

    def __getitem__(self, idx):
        #return the item at the index idx
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        #return the features and the label
        return self.df.iloc[idx, :-1], self.df.iloc[idx, -1]

# Create the train and test datasets and corresponding dataloaders.
train_dataset = VoiceDataset(r'C:/Users/maeva/Downloads/voice.csv')
test_dataset = VoiceDataset(r'C:/Users/maeva/Downloads/voice.csv')

    #create dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

