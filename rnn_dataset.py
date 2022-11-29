from torch.utils.data import Dataset, DataLoader
import torch

class SequenceDataset(Dataset):
    def __init__(self, data, input_col_1, input_col_2, task_cols):
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.input_col_1 = input_col_1
        self.input_col_2 = input_col_2
        self.task_cols = task_cols

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        # return x, y, s - s is the length of the input sequence
        
        x1 = self.data.loc[idx,self.input_col_1]
        x2 = self.data.loc[idx,self.input_col_2]
        s = len(x1)
        y = self.data.loc[idx,self.task_cols].tolist()
        
        return x1, x2, y, s

# Prepare dataloaders
def padding_tensor(arr, maxlen, dtype):
    padded_sess = torch.ones(len(arr), maxlen, dtype=dtype) * 0
    
    for i in range(len(arr)):
        padded_sess[i, :len(arr[i])] = torch.tensor(arr[i])
    
    return padded_sess 

def essay_collate_fn(batch):
    inputs_1 = [x[0] for x in batch]
    inputs_2 = [x[1] for x in batch]
    entity_seq_len = [x[3] for x in batch]
    
    labels = [x[2] for x in batch]

    maxlen = max(entity_seq_len)
        
    padded_inputs_1 = padding_tensor(inputs_1, maxlen, dtype=torch.int32)
    padded_inputs_2 = padding_tensor(inputs_2, maxlen, dtype=torch.int32)

    return (padded_inputs_1, padded_inputs_2, torch.tensor(entity_seq_len)), torch.tensor(labels)

