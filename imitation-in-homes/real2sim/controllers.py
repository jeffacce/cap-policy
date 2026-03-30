import numpy as np

class PID:
    def __init__(self, Kp, Ki, Kd, dt, out_limits=(None,None)):
        self.Kp, self.Ki, self.Kd, self.dt = Kp, Ki, Kd, dt
        self.integral = 0.0
        self.prev_error = 0.0
        self.out_min, self.out_max = out_limits

    def __call__(self, error):
        P = self.Kp * error
        self.integral += error * self.dt
        I = self.Ki * self.integral
        D = self.Kd * (error - self.prev_error) / self.dt
        u = P + I + D
        if self.out_min is not None: u = max(self.out_min, u)
        if self.out_max is not None: u = min(self.out_max, u)
        self.prev_error = error
        return u
