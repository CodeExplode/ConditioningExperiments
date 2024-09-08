import torch
import torch.nn as nn

# a small network to alter the conditioning vectors for the bos and eos padding, which are taken from a blank prompt encoded with CLIP, and seemingly need to be altered to fit the other conditioning vectors
class ConditioningAdjustmentNetwork(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=768):
        super(ConditioningAdjustmentNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        #self.relu = nn.ReLU() # don't think this is ideal, as conditioning vectors do have negative values
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        
        # attempt to have the data pass through unchanged in the default version of the network, initializing weights as an identity matrix and zero biases
        nn.init.eye_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        
        nn.init.eye_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    # conditioning: (batch_size, 77, 768) input conditioning vectors
    # mask: (batch_size, 77) marking the positions of BOS and EOS vectors to be altered
    def forward(self, conditioning, mask):
        adjusted_conditioning = self.fc1(conditioning)
        #adjusted_conditioning = self.relu(adjusted_conditioning)
        adjusted_conditioning = self.fc2(adjusted_conditioning)
        
        mask = mask.unsqueeze(-1) # (batch_size, 77, 1)
        adjusted_conditioning = conditioning * (1 - mask) + adjusted_conditioning * mask
        
        return adjusted_conditioning