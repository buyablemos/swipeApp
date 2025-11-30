import torch
import torch.nn.functional as F
import numpy as np
import json
from train_lightning import EEGTask
from data_preprocessing import process_sample, TARGET_LENGTH
import glob
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

VOTING_TYPE = 'hierarchical'  # Options: 'hard' or 'soft' or 'hierarchical'
THRESHOLD = 0.99

class EnsembleSystem:
    def __init__(self, blink_ckpt, right_ckpt, left_ckpt):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            
        print(f"Loading models on {self.device}...")
        
        self.model_blink = self._load_model(blink_ckpt)
        self.model_right = self._load_model(right_ckpt)
        self.model_left = self._load_model(left_ckpt)
        
        self.active_indices = {
            'blink': 0,
            'left': 0,
            'right': 1
        }

    def _load_model(self, ckpt_path):
        try:
            model = EEGTask.load_from_checkpoint(ckpt_path)
            model.to(self.device)
            model.eval()
            model.freeze()
            return model
        except Exception as e:
            print(f"Error loading checkpoint {ckpt_path}: {e}")
            raise e

    def predict(self, signal_tensor):
        signal_tensor = signal_tensor.to(self.device)
        with torch.no_grad():
            logits_b = self.model_blink(signal_tensor)
            probs_b = F.softmax(logits_b, dim=1)[0]
            logits_r = self.model_right(signal_tensor)
            probs_r = F.softmax(logits_r, dim=1)[0]
            logits_l = self.model_left(signal_tensor)
            probs_l = F.softmax(logits_l, dim=1)[0]
        p_blink = probs_b[self.active_indices['blink']].item()
        p_right = probs_r[self.active_indices['right']].item()
        p_left = probs_l[self.active_indices['left']].item()
        candidates = {
            'blink': p_blink,
            'right': p_right,
            'left': p_left
        }
        valid_candidates = {k: v for k, v in candidates.items() if v > THRESHOLD}

        if VOTING_TYPE == 'hierarchical':
            if candidates['blink'] > 0.85:
                return 'blink', candidates['blink']
            elif candidates['left'] > THRESHOLD:
                return 'left', candidates['left']
            elif candidates['right'] > THRESHOLD:
                return 'right', candidates['right']
            else:
                return 'none', max(1 - p_blink, 1 - p_right, 1 - p_left)
        elif VOTING_TYPE == 'hard':
            if not valid_candidates or (len(valid_candidates) > 1):
                return 'none', max(1 - p_blink, 1 - p_right, 1 - p_left)
            (best_class, best_conf), = valid_candidates.items()
            return best_class, best_conf
        elif VOTING_TYPE == 'soft':
            if not valid_candidates:
                return 'none', max(1 - p_blink, 1 - p_right, 1 - p_left)
            best_class = max(valid_candidates, key=valid_candidates.get)
            best_conf = valid_candidates[best_class]
            return best_class, best_conf
        else:
            raise ValueError(f"Unknown VOTING_TYPE: {VOTING_TYPE}")


def run_test_on_file(json_file, system):
    print(f"Running ensemble test on {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    labels = ['blink', 'left', 'right', 'none']
    
    for sample in data:
        label = sample.get('label')
        if label not in labels:
            continue
        proc = process_sample(sample, TARGET_LENGTH)
        if not proc:
            continue
        channels = [proc['data']['O1'], proc['data']['O2']]
        signal = np.array(channels)
        tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
        pred_label, conf = system.predict(tensor)
        y_true.append(label)
        y_pred.append(pred_label)
        if pred_label == label:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0
    print(f"Accuracy: {acc:.2%} ({correct}/{total})")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    ckpt_dir = "neuro_hackathon_eeg"
    ckpt_files = sorted(
        glob.glob(os.path.join(ckpt_dir, "**", "*.ckpt"), recursive=True),
        key=os.path.getmtime,
        reverse=True
    )
    if len(ckpt_files) < 3:
        raise RuntimeError("Not enough checkpoint files found in neuro_hackathon_eeg.")
    CKPT_BLINK = ckpt_files[2]
    print(f"Blink: {CKPT_BLINK}")
    CKPT_LEFT = ckpt_files[1]
    print(f"Left: {CKPT_LEFT}")
    CKPT_RIGHT = ckpt_files[0]
    print(f"Right: {CKPT_RIGHT}")
    print("WARNING: Please update checkpoint paths in ensemble_system.py before running!")
    system = EnsembleSystem(CKPT_BLINK, CKPT_RIGHT, CKPT_LEFT)
    run_test_on_file('test_data.json', system)
