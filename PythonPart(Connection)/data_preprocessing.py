import json
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import collections

# --- Configuration ---
INPUT_FILE = 'eeg_data_final.json'
TRAIN_FILE = 'train_data.json'
TEST_FILE = 'test_data.json'

# Sampling rate is critical for filtering. 
# Assuming 250 Hz based on typical Muse/EEG data context, but adjustable here.
SAMPLING_RATE = 250.0 

# Filter parameters
LOWCUT = 0.1   # Hz
HIGHCUT = 50.0 # Hz
FILTER_ORDER = 4

# Length normalization
TARGET_LENGTH = 500 # samples (~2 seconds at 250Hz)

# Augmentation
AUGMENT_DATA = True

# Outlier Removal
REMOVE_OUTLIERS = True
OUTLIER_CONTAMINATION = 0.1 # Remove 10% most different samples per class


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # Check for valid cutoff frequencies
    if low <= 0: low = 0.001
    if high >= 1: high = 0.999
    
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    # filtfilt requires the data length to be greater than the filter order * padlen
    # Default padlen is 3 * max(len(a), len(b)). 
    # For order 4, len(a) is 9. 3*9 = 27. So we need at least ~30 samples.
    if len(data) < 30: 
        return data # Return original if too short to filter safely
        
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    try:
        y = filtfilt(b, a, data)
        return y
    except Exception as e:
        print(f"Warning: Filtering failed for signal of length {len(data)}. Error: {e}")
        return data

def process_sample(sample, target_length):
    raw_data = sample.get('data', {})
    label = sample.get('label', 'unknown')
    
    # Ensure we have the channels we expect
    if 'O1' not in raw_data or 'O2' not in raw_data:
        return None
        
    o1 = np.array(raw_data['O1'])
    o2 = np.array(raw_data['O2'])
    
    # Skip empty samples
    if len(o1) == 0 or len(o2) == 0:
        return None

    # 1. Apply Bandpass Filter
    # Filtering should be done on the continuous signal BEFORE cutting/padding to avoid edge artifacts
    o1_filtered = bandpass_filter(o1, LOWCUT, HIGHCUT, SAMPLING_RATE, FILTER_ORDER)
    o2_filtered = bandpass_filter(o2, LOWCUT, HIGHCUT, SAMPLING_RATE, FILTER_ORDER)

    # 2. Normalize (Z-score) per channel
    # We normalize before padding so that the mean is 0, making zero-padding appropriate.
    def normalize(x):
        m = np.mean(x)
        s = np.std(x)
        return (x - m) / (s + 1e-8) if s > 0 else x

    o1_norm = normalize(o1_filtered)
    o2_norm = normalize(o2_filtered)
    
    # 3. Length Normalization (Cut or Pad)
    current_len = len(o1_norm)
    
    if current_len >= target_length:
        # Truncate (take the first target_length samples)
        o1_final = o1_norm[:target_length]
        o2_final = o2_norm[:target_length]
    else:
        # Pad
        pad_width = target_length - current_len
        # Pad with zeros at the end. 
        # Since we filtered and normalized, the mean is 0, so zero-padding is appropriate.
        o1_final = np.pad(o1_norm, (0, pad_width), 'constant', constant_values=0)
        o2_final = np.pad(o2_norm, (0, pad_width), 'constant', constant_values=0)
        
    return {
        "data": {
            "O1": o1_final.tolist(),
            "O2": o2_final.tolist()
        },
        "label": label
    }

def generate_augmentations(sample):
    """
    Generates augmented versions of a single sample.
    Returns a list of new samples (dicts).
    """
    o1 = np.array(sample['data']['O1'])
    o2 = np.array(sample['data']['O2'])
    label = sample['label']
    
    augmented = []
    
    # 1. Gaussian Noise
    # Since data is Z-normalized (std approx 1), noise_level=0.1 is 10% of signal power
    noise_level = 0.1
    noise1 = np.random.normal(0, noise_level, o1.shape)
    noise2 = np.random.normal(0, noise_level, o2.shape)
    augmented.append({
        "data": {
            "O1": (o1 + noise1).tolist(),
            "O2": (o2 + noise2).tolist()
        },
        "label": label
    })
    
    # 2. Amplitude Scaling
    # Simulates different signal strengths / impedance changes
    scale = np.random.uniform(0.8, 1.2)
    augmented.append({
        "data": {
            "O1": (o1 * scale).tolist(),
            "O2": (o2 * scale).tolist()
        },
        "label": label
    })
    
    # 3. Time Shift
    # Shift signal left or right and pad with zeros
    shift = np.random.randint(-30, 30) # +/- ~0.1s
    def shift_arr(arr, s):
        res = np.roll(arr, s)
        if s > 0:
            res[:s] = 0
        elif s < 0:
            res[s:] = 0
        return res
        
    augmented.append({
        "data": {
            "O1": shift_arr(o1, shift).tolist(),
            "O2": shift_arr(o2, shift).tolist()
        },
        "label": label
    })
    
    return augmented

def filter_outliers(data):
    """
    Removes outliers from each class using Isolation Forest.
    """
    print(f"Removing outliers using Isolation Forest (contamination={OUTLIER_CONTAMINATION})...")
    cleaned_data = []
    
    # Group by label
    data_by_label = collections.defaultdict(list)
    for sample in data:
        data_by_label[sample['label']].append(sample)
        
    for label, samples in data_by_label.items():
        # Skip if too few samples to fit
        if len(samples) < 10:
            print(f"Skipping outlier removal for class '{label}' (too few samples: {len(samples)})")
            cleaned_data.extend(samples)
            continue
            
        # Prepare feature matrix: Flatten O1 and O2
        X = []
        for s in samples:
            # Concatenate channels to form a single feature vector
            features = np.concatenate([np.array(s['data']['O1']), np.array(s['data']['O2'])])
            X.append(features)
        
        X = np.array(X)
        
        # Fit Isolation Forest
        # n_jobs=-1 uses all cores
        clf = IsolationForest(contamination=OUTLIER_CONTAMINATION, random_state=42, n_jobs=-1)
        preds = clf.fit_predict(X) # 1 for inlier, -1 for outlier
        
        # Keep inliers
        inliers = [s for s, p in zip(samples, preds) if p == 1]
        outliers_count = len(samples) - len(inliers)
        print(f"Class '{label}': Removed {outliers_count} outliers from {len(samples)} samples.")
        
        cleaned_data.extend(inliers)
        
    return cleaned_data

def main():
    print(f"Loading data from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Total raw samples: {len(raw_data)}")
    
    # Analyze lengths before processing
    lengths = [len(s['data']['O1']) for s in raw_data if 'data' in s and 'O1' in s['data']]
    if lengths:
        print(f"Sample lengths - Min: {min(lengths)}, Max: {max(lengths)}, Mean: {np.mean(lengths):.1f}, Median: {np.median(lengths)}")
    
    processed_data = []
    labels = []
    
    print(f"Processing samples (Filter {LOWCUT}-{HIGHCUT}Hz, Target Length {TARGET_LENGTH})...")
    
    for sample in raw_data:
        proc = process_sample(sample, TARGET_LENGTH)
        if proc:
            processed_data.append(proc)
            labels.append(proc['label'])
            
    print(f"Successfully processed {len(processed_data)} samples.")
    
    # --- Outlier Removal ---
    if REMOVE_OUTLIERS:
        processed_data = filter_outliers(processed_data)
        print(f"Samples after outlier removal: {len(processed_data)}")
        # Update labels to match filtered data
        labels = [d['label'] for d in processed_data]
    # -----------------------

    # Check class distribution
    counter = collections.Counter(labels)
    print("Class distribution:", dict(counter))
    
    # Split into Train and Test
    # IMPORTANT: Split BEFORE augmentation to prevent data leakage!
    # We don't want augmented versions of test samples in the training set.
    try:
        train_data, test_data = train_test_split(
            processed_data, 
            test_size=0.2, 
            stratify=labels, 
            random_state=42, 
            shuffle=True
        )
    except ValueError as e:
        print(f"Error during split (probably too few samples for a class): {e}")
        # Fallback without stratify if a class has only 1 sample
        train_data, test_data = train_test_split(
            processed_data, 
            test_size=0.2, 
            random_state=42, 
            shuffle=True
        )

    print(f"Initial Train set size: {len(train_data)}")
    print(f"Test set size: {len(test_data)}")
    
    # Augment Training Data
    if AUGMENT_DATA:
        print("Augmenting training data...")
        augmented_train = []
        for sample in train_data:
            # Keep original
            augmented_train.append(sample)
            # Generate new ones
            new_samples = generate_augmentations(sample)
            augmented_train.extend(new_samples)
        
        train_data = augmented_train
        print(f"Augmented Train set size: {len(train_data)}")
    
    # Save to files
    print(f"Saving to {TRAIN_FILE} and {TEST_FILE}...")
    with open(TRAIN_FILE, 'w') as f:
        json.dump(train_data, f)
        
    with open(TEST_FILE, 'w') as f:
        json.dump(test_data, f)
        
    print("Done.")

if __name__ == "__main__":
    main()
