import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, spectrogram

# Load EEG data from JSON
with open('eeg_data_test.json', 'r') as f:
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

# Create subplots for each index
num_indexes = len(data)
fig, axes = plt.subplots(num_indexes, 2, figsize=(16, 6 * num_indexes))

if num_indexes == 1:
    axes = [axes]  # Ensure axes is iterable for a single subplot

for index, entry in enumerate(data):
    # Extract channel data
    channel_O1 = np.array(entry['data']['O1'])
    channel_O2 = np.array(entry['data']['O2'])
    label = entry['label']

    # Apply bandpass filter
    filtered_O1 = bandpass_filter(channel_O1, lowcut, highcut, sampling_rate, order=4)
    filtered_O2 = bandpass_filter(channel_O2, lowcut, highcut, sampling_rate, order=4)

    # Plot spectrograms
    f1, t1, Sxx1 = spectrogram(filtered_O1, fs=sampling_rate)
    f2, t2, Sxx2 = spectrogram(filtered_O2, fs=sampling_rate)

    ax1 = axes[index][0]
    ax2 = axes[index][1]

    im1 = ax1.pcolormesh(t1, f1, 10 * np.log10(Sxx1), shading='gouraud')
    ax1.set_title(f'Index {index} - O1 Spectrogram - Label: {label}')
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_xlabel('Time (s)')
    fig.colorbar(im1, ax=ax1)

    im2 = ax2.pcolormesh(t2, f2, 10 * np.log10(Sxx2), shading='gouraud')
    ax2.set_title(f'Index {index} - O2 Spectrogram - Label: {label}')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlabel('Time (s)')
    fig.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.show()
