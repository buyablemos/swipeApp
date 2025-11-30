import argparse
import brainaccess_board as bb
from brainaccess_board.message_queue import BoardControl
from time import sleep
import json

def get_channel_data(rawdata, num_samples, channel_names=["O1", "O2"]):
    data = {}
    for channel in channel_names:
        data[channel] = rawdata.get_data(picks=[channel])[0][:num_samples]
    return data

def save_data_json(data, label="none", file_name="eeg_data.json"):
    eeg_entry = {
        "data": {channel: data[channel].tolist() for channel in data} if isinstance(data, dict) else data,
        "label": label
    }

    try:
        try:
            with open(file_name, "r") as f:
                existing_data = json.load(f)
        except FileNotFoundError:
            existing_data = []

        existing_data.append(eeg_entry)

        with open(file_name, "w") as f:
            json.dump(existing_data, f, indent=4)
        print(f"Data saved to {file_name}")
    except Exception as e:
        print(f"Error saving data to JSON: {e}")

def main():
    parser = argparse.ArgumentParser(description="EEG Data Collection Script")
    parser.add_argument("--time", type=float, required=True, help="Recording time in seconds")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of samples to record (-1 saves all recorded samples)")
    parser.add_argument("--label", type=str, required=True, help="Label for the recorded data")
    parser.add_argument("--num_rounds", type=int, default=25, help="Number of recording rounds")

    args = parser.parse_args()
    time = args.time
    num_samples = args.num_samples
    label = args.label
    num_rounds = args.num_rounds

    db, status = bb.db_connect()
    if status:
        board_control = BoardControl()
        response = board_control.get_commands()

        board_control.command(response["data"]["stop_recording"])
        for i in range(num_rounds):
            board_control.command(response["data"]["start_recording"])
            print(f"\n \n \n {i} START RECORDING \n \n \n")

            sleep(time)
            print(f"\n \n \n {i} STOP RECORDING \n \n \n")
            device = db.get_mne()
            rawdata = device[next(iter(device))]

            if num_samples == -1:
                data = rawdata.get_data(picks=["O1", "O2"])
                num_samples_iter = min(data[0].shape[0], data[1].shape[0])
                
            data = get_channel_data(rawdata, num_samples_iter, channel_names=["O1", "O2"])
            if any(len(data[channel]) < num_samples_iter for channel in data):
                print("Data length less than specified number of samples for one or more channels")
                board_control.command(response["data"]["stop_recording"])
                sleep(2) 
                continue
            save_data_json(data, label=label, file_name="eeg_data.json")
            # sleep(2)   

            board_control.command(response["data"]["stop_recording"])

if __name__ == "__main__":
    main()