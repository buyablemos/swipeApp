import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import wandb

# Import models
from simple_cnn import SimpleCNN
# from EEGNet import EEGClassifier as EEGNetModel
# from EEGTransformer import EEGClassifier as EEGTransformerModel
# from lstm import LSTMModel


class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_data_from_json(json_file):
    """Loads preprocessed data from JSON."""
    with open(json_file, 'r') as f:
        data = json.load(f)

    X = []
    y = []
    
    for entry in data:
        # Data is already preprocessed (O1, O2 lists)
        channels = []
        # Ensure consistent order of channels
        for ch_name in sorted(entry['data'].keys()): 
            channels.append(entry['data'][ch_name])
        
        signal = np.array(channels) # Shape: (n_channels, n_samples)
        X.append(signal)
        y.append(entry['label'])

    return np.array(X), np.array(y)

# --- 2. Lightning Module (The Task) ---

class EEGTask(pl.LightningModule):
    def __init__(self, model_name, model_config, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_name = model_name
        self.model = self._build_model(model_name, model_config)
        self.loss_fn = nn.CrossEntropyLoss()

    def _build_model(self, name, config):
        if name == 'cnn':
            return SimpleCNN(
                input_channels=config['input_channels'],
                input_length=config['input_length'],
                num_classes=config['num_classes']
            )
        elif name == 'eegnet':
            return EEGNetModel(
                num_channels=config['input_channels'],
                num_classes=config['num_classes'],
                time_points=config['input_length']
            )
        elif name == 'transformer':
            return EEGTransformerModel(
                num_channels=config['input_channels'],
                num_classes=config['num_classes'],
                window_size=config['input_length'],
                d_model=config.get('d_model', 64),
                nhead=config.get('nhead', 4),
                num_layers=config.get('num_layers', 2)
            )
        elif name == 'lstm':
            return LSTMModel(
                input_channels=config['input_channels'],
                num_classes=config['num_classes'],
                hidden_size=config.get('hidden_size', 64),
                num_layers=config.get('num_layers', 2),
                dropout=config.get('dropout', 0.5)
            )
        else:
            raise ValueError(f"Unknown model name: {name}")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=0.01)

# --- 3. Main Execution ---

if __name__ == "__main__":
    # --- CONFIGURATION ---
    TRAIN_FILE = "train_data.json"
    TEST_FILE = "test_data.json"
    BATCH_SIZE = 16
    MAX_EPOCHS = 100
    LEARNING_RATE = 1e-3
    
    # Select Model: 'cnn', 'eegnet', 'transformer', 'lstm'
    SELECTED_MODEL = 'cnn' 
    
    # Model specific hyperparameters (optional overrides)
    MODEL_PARAMS = {
        'transformer': {'d_model': 64, 'nhead': 4, 'num_layers': 2},
        'lstm': {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.5}
    }
    # ---------------------
    
    print(f"Loading training data from {TRAIN_FILE}...")
    try:
        X_train, y_train_raw = load_data_from_json(TRAIN_FILE)
        print(f"Loading test data from {TEST_FILE}...")
        X_test, y_test_raw = load_data_from_json(TEST_FILE)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please run data_preprocessing.py first.")
        exit(1)

    # Encode labels
    # Fit encoder on ALL labels to ensure consistency and handle all classes
    all_labels = np.concatenate([y_train_raw, y_test_raw])
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    
    y_train = label_encoder.transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)
    
    print(f"Classes: {label_encoder.classes_}")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Create Datasets
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_test, y_test) # Using test set as validation here
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Prepare Model Config
    input_channels = X_train.shape[1]
    input_length = X_train.shape[2]
    num_classes = len(label_encoder.classes_)
    
    config = {
        'input_channels': input_channels,
        'input_length': input_length,
        'num_classes': num_classes
    }
    
    # Add specific params if any
    if SELECTED_MODEL in MODEL_PARAMS:
        config.update(MODEL_PARAMS[SELECTED_MODEL])
    
    print(f"Initializing model: {SELECTED_MODEL} with config: {config}")
    
    model = EEGTask(
        model_name=SELECTED_MODEL,
        model_config=config,
        learning_rate=LEARNING_RATE
    )
    
    # Logger
    wandb_logger = WandbLogger(project="neuro_hackathon_eeg", name=f"{SELECTED_MODEL}_v7")
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        logger=wandb_logger,
        accelerator="auto",
        devices=1,
        log_every_n_steps=2
    )
    
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    print("Training complete.")
    wandb.finish()
