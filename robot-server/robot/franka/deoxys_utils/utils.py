import time
import numpy as np
from typing import Tuple

from .messages import ControllerState


def notify_component_start(component_name):
    print("***************************************************************")
    print("     Starting {} component".format(component_name))
    print("***************************************************************")


class FrequencyTimer(object):
    def __init__(self, frequency_rate):
        self.time_available = 1e9 / frequency_rate

    def start_loop(self):
        self.start_time = time.time_ns()

    def check_time(self, frequency_rate):
        # if prev_check_time variable doesn't exist, create it
        if not hasattr(self, "prev_check_time"):
            self.prev_check_time = self.start_time

        curr_time = time.time_ns()
        if (curr_time - self.prev_check_time) > 1e9 / frequency_rate:
            self.prev_check_time = curr_time
            return True
        return False

    def end_loop(self):
        wait_time = self.time_available + self.start_time

        while time.time_ns() < wait_time:
            continue


def parse_controller_state(controller_state_string: str) -> ControllerState:

    left_data, right_data = controller_state_string.split("|")

    left_data = left_data.split(";")[1:-1]
    right_data = right_data.split(";")[1:-1]

    def parse_bool(val: str) -> bool:
        return val.split(":")[1].lower().strip() == "true"

    def parse_float(val: str) -> float:
        return float(val.split(":")[1])

    def parse_list_float(val: str) -> np.ndarray:
        return np.array(list(map(float, val.split(":")[1].split(","))))

    def parse_section(data: list) -> Tuple:
        return (
            # Buttons
            parse_bool(data[0]),
            parse_bool(data[1]),
            parse_bool(data[2]),
            parse_bool(data[3]),
            # Triggers
            parse_float(data[4]),
            parse_float(data[5]),
            # Thumbstick
            parse_list_float(data[6]),
            # Pose
            parse_list_float(data[7]),
            parse_list_float(data[8]),
        )

    left_parsed = parse_section(left_data)
    right_parsed = parse_section(right_data)

    return ControllerState(time.time(), *left_parsed, *right_parsed)
