# page_scrolling.py
import brainaccess_board as bb
from brainaccess_board.message_queue import BoardControl
from time import sleep
import time
import numpy as np

import torch
from ensemble_system import EnsembleSystem
from data_preprocessing import process_sample, TARGET_LENGTH
from selenium.webdriver.common.by import By

import os

def get_one_channel_data(rawdata, channel_name="O1"):
    data = rawdata.get_data(picks=[channel_name])[0]
    return data[-500:]


# ---------------------
# LOAD ENSEMBLE SYSTEM
# ---------------------
CKPT_BLINK = "neuro_hackathon_eeg/blink/checkpoints/epoch=49-step=500.ckpt"
CKPT_RIGHT = "neuro_hackathon_eeg/right/checkpoints/epoch=49-step=500.ckpt"
CKPT_LEFT  = "neuro_hackathon_eeg/left/checkpoints/epoch=49-step=500.ckpt"

system = EnsembleSystem(CKPT_BLINK, CKPT_RIGHT, CKPT_LEFT)

# ---------------------
# SELENIUM SETUP
# ---------------------
from selenium import webdriver
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(options=options)


url = "http://localhost:5173"
driver.get(url)
time.sleep(1)

viewport_height = driver.execute_script("return window.innerHeight")

body = driver.find_element(By.TAG_NAME, "body")

def press_left():
    """Wciśnij 'a' - Pass/Odrzuć"""
    body.send_keys('a')
    print("⬅️ Wciśnięto 'a' - Pass")


def press_right():
    """Wciśnij 'd' - Like/Ulubione"""
    body.send_keys('d')
    print("➡️ Wciśnięto 'd' - Like")


def press_up():
    """Wciśnij 'w' - Koszyk"""
    body.send_keys('w')
    print("⬆️ Wciśnięto 'w' - Koszyk")



# ---------------------
# CONNECT TO BRAINACCESS
# ---------------------
db, status = bb.db_connect()
if status:
    board_control = BoardControl()
    response = board_control.get_commands()

    board_control.command(response["data"]["stop_recording"])
    i = 0

    while True:
        if i == 0:
            board_control.command(response["data"]["start_recording"])

        sleep(0.5)

        device = db.get_mne()
        rawdata = device[next(iter(device))]
        o1 = get_one_channel_data(rawdata, "O1")
        o2 = get_one_channel_data(rawdata, "O2")

        if len(o1) < 500 or len(o2) < 500:
            print("Data length < 500 samples")
            continue

        # Prepare tensor for ensemble (1,2,500)
        sample = {
            "data": {
                "O1": o1,
                "O2": o2
            },
            "label": "unknown"
        }

        processed = process_sample(sample, TARGET_LENGTH)

        if processed is None:
            print("Skipping: preprocessing returned None")
            continue

        o1_final = np.array(processed["data"]["O1"])
        o2_final = np.array(processed["data"]["O2"])

        # final shape: (1,2,TARGET_LENGTH)
        signal = np.stack([o1_final, o2_final], axis=0)
        tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)

        # -------------------------
        # ENSEMBLE PREDICTION
        # -------------------------
        pred_class, conf = system.predict(tensor)

        # PRINT FINAL DECISION IN TERMINAL
        print(f"ENSEMBLE decision: {pred_class} (confidence={conf:.3f})")

        # -------------------------
        # MAP ENSEMBLE → SCROLLING
        # -------------------------
        if pred_class == "blink":
            press_up()
            sleep(2)

        elif pred_class == "left":
            press_left()
            sleep(2)
        elif pred_class == "right":
            press_right()
            sleep(2)
        else:
            print("→ none → no action")
            sleep(0.5)

        if i == 0:
            board_control.command(response["data"]["stop_recording"])

        i = (i + 1) % 100
