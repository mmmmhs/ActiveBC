import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

class BehaviorCloningModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BehaviorCloningModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax() # output a probability distribution
        )
    
    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        probs = self.model(x)
        return torch.argmax(probs, dim=-1)
    
    def entropy(self, x):
        probs = self.model(x)
        return -torch.sum(probs * torch.log(probs), dim=-1)

class Trainer:
    def __init__(self, model, lr, device=None, load_path=None):
        """
        Initialize the trainer with a model and optional device.
        Args:
            model: PyTorch model for binary classification.
            device: torch.device to use for training (e.g., 'cpu' or 'cuda').
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        if load_path:
            self.load_model(load_path)

    def train(self, X, Y, epochs=10, batch_size=32, patience=5, interval=10, save_path='model.pth'):
        """
        Train the model with early stopping.
        Args:
            X: Input features (torch.Tensor).
            Y: Target labels (torch.Tensor).
            epochs: Maximum number of epochs to train.
            batch_size: Batch size for training.
            learning_rate: Learning rate for optimizer.
            patience: Number of epochs to wait for improvement before stopping.
            save_path: Path to save the best model.
        Returns:
            dict: Training history with loss values.
        """
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
        X_train, X_val = torch.tensor(X_train).float(), torch.tensor(X_val).float()
        Y_train, Y_val = torch.tensor(Y_train).long(), torch.tensor(Y_val).long()
        train_dataset = TensorDataset(X_train, Y_train)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()

        best_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': []}

        for epoch in tqdm(range(epochs)):
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_Y in train_dataloader:
                batch_X, batch_Y = batch_X.to(self.device), batch_Y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_Y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * batch_X.size(0)

            train_loss /= len(train_dataloader.dataset)
            history['train_loss'].append(train_loss)

            # Validation phase
            val_loss = self.evaluate(X_val, Y_val)
            history['val_loss'].append(val_loss)
            if (epoch + 1) % interval == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                # save the best model and optimizer state
                torch.save({'model_state_dict': self.model.state_dict(), 
                            'optimizer_state_dict': self.optimizer.state_dict()}, save_path)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

        return history

    def load_model(self, load_path):
        """
        Load a trained model from file.
        Args:
            load_path: Path to the saved model file.
        """
        self.model.load_state_dict(torch.load(load_path)['model_state_dict'])
        self.optimizer.load_state_dict(torch.load(load_path)['optimizer_state_dict'])
        print("Model loaded from", load_path)

    def evaluate(self, X, Y):
        """
        Evaluate the model on a dataset.
        Args:
            X: Input features (torch.Tensor).
            Y: Target labels (torch.Tensor).
        Returns:
            float: Loss on the dataset.
        """
        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch_X, batch_Y in dataloader:
                batch_X, batch_Y = batch_X.to(self.device), batch_Y.to(self.device)
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_Y)
                total_loss += loss.item() * batch_X.size(0)

        total_loss /= len(dataloader.dataset)
        # print(f"Evaluation Loss: {total_loss:.4f}")
        return total_loss

    def predict(self, X):
        """
        Predict the labels for a given input.
        Args:
            X: Input features (torch.Tensor).
        Returns:
            torch.Tensor: Predicted labels.
        """
        X = torch.tensor(X).float()
        dataset = TensorDataset(X)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch_X in dataloader:
                batch_X = batch_X[0].to(self.device)
                outputs = self.model.predict(batch_X)
                predictions.append(outputs)

        return torch.cat(predictions).to('cpu').numpy()
    
    def predict_one(self, X):
        """
        Predict the label for a single input.
        Args:
            X: Input features (torch.Tensor).
        Returns:
            int: Predicted label.
        """
        self.model.eval()
        X = torch.tensor(X).float()
        with torch.no_grad():
            X = X.to(self.device)
            output = self.model.predict(X)
        return output.item()

    def get_entropy(self, X):
        """
        Get the entropy of the model predictions.
        Args:
            X: Input features (torch.Tensor).
        Returns:
            torch.Tensor: Entropy values.
        """
        X = torch.tensor(X).float()
        dataset = TensorDataset(X)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        self.model.eval()
        entropies = []

        with torch.no_grad():
            for batch_X in tqdm(dataloader):
                batch_X = batch_X[0].to(self.device)
                outputs = self.model.entropy(batch_X)
                entropies.append(outputs)

        return torch.cat(entropies).to('cpu').numpy()
