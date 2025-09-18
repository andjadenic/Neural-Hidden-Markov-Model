import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class Small_dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, id):
        return (self.x[id,:], self.y[id])


class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=2,
                             out_features=3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=3,
                             out_features=2)

    def forward(self, x):
        '''
        x: (Nb, 2) tensor
        '''
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

if __name__ == "__main__":

    # Data
    x_data = torch.tensor([
        [0.1, 0.2],
        [0.9, 0.1],
        [0.4, 0.8],
        [0.7, 0.9],
        [0.2, 0.6],
        [0.8, 0.5],
        [0.3, 0.4],
        [0.6, 0.7]
    ], dtype=torch.float)
    y_data = y_data = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1], dtype=torch.long)

    # Wrap data in a Dataset and DataLoader
    dataset = Small_dataset(x_data, y_data)
    dataloader = DataLoader(dataset,
                            batch_size=2,
                            shuffle=True)

    # Initialize model, loss and optimizer
    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=.01)

    # Try forward pass for single input
    '''
    i = torch.tensor([.2, .3])
    print(model(i))
    '''

    # Train the model
    '''N_epochs = 50
    for epoch in range(N_epochs):
        total_loss = 0  # Total loss in current epoch
        for batch_x, batch_y in dataloader:
            # Forward pass
            batch_p = model(batch_x) # model's prediction

            # Calculate the loss
            batch_loss = criterion(batch_p, batch_y)

            # Backward pass
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss
        print(f'{epoch=}, : {total_loss=}')

    # Save model parameters
    torch.save(model.state_dict(), "simple_nn.pth")
    print("Model parameters saved to simple_nn.pth")'''

    # Load model parameters
    trained_model = SimpleNN()
    trained_model.load_state_dict(torch.load("simple_nn.pth"))
    trained_model.eval()

    # Test predictions with loaded model
    test_x = torch.tensor([
        [0.2, 0.3],
        [0.8, 0.2],
        [0.5, 0.9]
    ], dtype=torch.float32)

    test_p = trained_model(test_x)
    print(test_p)
    test_p = test_p.argmax(dim=1)
    print(test_p)
