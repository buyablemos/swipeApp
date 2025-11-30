import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Load EEG data from JSON
with open('eeg_data_analysis.json', 'r') as f:
    data = json.load(f)

sampling_rate = 250  # Hz

# Design a bandpass Butterworth filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Bandpass filter parameters
lowcut = 0.1  # Hz
highcut = 50.0  # Hz

# Create subplots for each index, excluding 'up' and 'jaw' labels
filtered_data = [entry for entry in data if entry['label'] not in ['up', 'jaw']]
num_indexes = len(filtered_data)
fig, axes = plt.subplots(
    num_indexes, 1,
    figsize=(12, 6 * num_indexes),
    constrained_layout=True
)

if num_indexes == 1:
    axes = [axes]  # Ensure axes is iterable for a single subplot

for index, entry in enumerate(filtered_data):
    # Extract channel data
    channel_O1 = np.array(entry['data']['O1'])
    channel_O2 = np.array(entry['data']['O2'])
    label = entry['label']

    # Apply bandpass filter
    filtered_O1 = bandpass_filter(channel_O1, lowcut, highcut, sampling_rate, order=4)
    filtered_O2 = bandpass_filter(channel_O2, lowcut, highcut, sampling_rate, order=4)

    # Time vector
    time = np.arange(len(channel_O1)) / sampling_rate

    # Plot filtered signals on the corresponding subplot
    ax = axes[index]
    ax.plot(time, filtered_O1, label='O1')
    ax.plot(time, filtered_O2, label='O2')
    ax.set_title(
        f'Index {index} - Label: {label}',
        fontsize=10,
        pad=15
    )
    ax.set_xlabel('Time (s)')

plt.tight_layout()
plt.subplots_adjust(top=0.93, hspace=0.7)
plt.show()
