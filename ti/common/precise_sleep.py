import time

def precise_sleep(dt: float, slack_time: float = 0.001, time_func=time.monotonic):
    """
    Performs a precise sleep for a specified duration using a combination of time.sleep and active spinning.

    Args:
        dt (float): The total duration to sleep, in seconds.
        slack_time (float, optional): The duration to actively spin (busy-wait) after sleeping, to achieve precision. Defaults to 0.001 seconds.
        time_func (function, optional): The time function used for measuring time. Defaults to time.monotonic.

    Description:
        This function first uses time.sleep to sleep for 'dt - slack_time' seconds, 
        allowing for low CPU usage during most of the sleep duration. Then, it actively spins 
        (busy-wait) for the remaining 'slack_time' to ensure precise wake-up. This hybrid approach 
        minimizes jitter caused by the non-deterministic nature of time.sleep.
    """
    t_start = time_func()
    if dt > slack_time:
        time.sleep(dt - slack_time)
    t_end = t_start + dt
    while time_func() < t_end:
        pass
    return

def precise_wait(t_end: float, slack_time: float = 0.001, time_func=time.monotonic):
    """
    Waits until a specified end time using a combination of time.sleep and active spinning for precision.

    Args:
        t_end (float): The target end time in seconds since a fixed point in the past (e.g., system start).
        slack_time (float, optional): The duration to actively spin (busy-wait) before reaching the target time, to achieve precision. Defaults to 0.001 seconds.
        time_func (function, optional): The time function used for measuring time. Defaults to time.monotonic.

    Description:
        This function calculates the remaining time to 't_end' and then uses a combination of 
        time.sleep and active spinning to wait until this target time. It sleeps for the bulk of 
        the remaining time, then actively spins for the final 'slack_time' duration, 
        thus ensuring precise timing.
    """
    t_start = time_func()
    t_wait = t_end - t_start
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time_func() < t_end:
            pass
    return
