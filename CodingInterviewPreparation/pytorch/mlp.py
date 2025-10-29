#%%
# * **Task** : Implement a Multi-Layer Perceptron from scratch (no `nn.Sequential`)
# * **Reference** : [PyTorch nn.Module Tutorial](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
# * **Code Template** :

# ```python

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__() 
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Your implementation here

    def forward(self, x):
        # Your implementation here
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
  
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # Your implementation here
        
  
    return running_loss / len(dataloader)

# Example usage
input_dim = 10
hidden_dim = 20
output_dim = 5

model = MLP(input_dim, hidden_dim, output_dim)
sample_input = torch.randn(100, input_dim)
output = model(sample_input)
print("Output shape:", output.shape)

#%%
# Example training loop (dummy data)
from torch.utils.data import DataLoader, TensorDataset 
batch_size = 16
dummy_inputs = torch.randn(100, input_dim)
dummy_targets = torch.randint(0, output_dim, (100,))
dataset = TensorDataset(dummy_inputs, dummy_targets)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
for epoch in range(5):
    avg_loss = train_one_epoch(model, dataloader, criterion, optimizer, device)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")



