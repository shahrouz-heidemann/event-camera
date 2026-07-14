"""
Event Camera (DVS) Simulation — a minimal, from-scratch implementation
=======================================================================

Goal
----
Event cameras (Dynamic Vision Sensors, DVS) don't output frames. Instead,
each pixel independently reports an "event" (x, y, t, polarity) whenever
the LOG intensity at that pixel changes by more than a threshold C:

    event fires when  | log(I(x,y,t)) - log(I(x,y,t_last)) | > C

This script:
1. Renders a synthetic video (a moving/rotating shape) frame by frame,
   since we don't have access to a real DVS sensor or downloadable
   dataset in this environment.
2. Converts that synthetic video into events using the standard DVS
   generative model above (the same principle used by simulators such
   as ESIM and v2e, and described in Gallego et al., "Event-based
   Vision: A Survey", IEEE TPAMI 2020).
3. Visualizes the result the way event-vision papers typically do:
   an "event frame" (blue = positive/brightness-increase events,
   red = negative/brightness-decrease events) next to the original
   grayscale frame.

This is intentionally simple (no noise model, no refractory period,
no real sensor data) — it's a first hands-on step to understand how
event generation works, not a research-grade simulator.

Dependencies: numpy, opencv-python, matplotlib (all already installed)
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Render a synthetic scene: a bright disc moving across a dark background
# ----------------------------------------------------------------------
WIDTH, HEIGHT = 240, 180
N_FRAMES = 60
CONTRAST_THRESHOLD = 0.15  # "C" in the DVS model — smaller = more sensitive


def render_frame(t: int) -> np.ndarray:
    """Render one grayscale frame (float32, range [0, 1]) at time step t."""
    img = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

    # A disc moving left-to-right and a smaller disc moving in a circle,
    # so the simulation contains both translational and curved motion.
    cx1 = int(20 + t * (WIDTH - 40) / N_FRAMES)
    cy1 = HEIGHT // 2
    cv2.circle(img, (cx1, cy1), 18, 1.0, -1)

    angle = 2 * np.pi * t / N_FRAMES
    cx2 = int(WIDTH * 0.75 + 30 * np.cos(angle))
    cy2 = int(HEIGHT * 0.3 + 30 * np.sin(angle))
    cv2.circle(img, (cx2, cy2), 10, 0.6, -1)

    # mild blur to avoid perfectly binary edges (more realistic gradients)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img


# ----------------------------------------------------------------------
# 2. Convert the frame sequence into events using the DVS log-intensity model
# ----------------------------------------------------------------------
def simulate_events(frames: list[np.ndarray], threshold: float):
    """
    Returns a list of events, each as (x, y, t, polarity).
    polarity = +1 for brightness increase, -1 for brightness decrease.
    """
    eps = 1e-3  # avoids log(0)
    log_ref = np.log(frames[0] + eps)  # reference log-intensity per pixel
    events = []

    for t in range(1, len(frames)):
        log_now = np.log(frames[t] + eps)
        diff = log_now - log_ref

        pos_y, pos_x = np.where(diff > threshold)
        neg_y, neg_x = np.where(diff < -threshold)

        for x, y in zip(pos_x, pos_y):
            events.append((x, y, t, 1))
            log_ref[y, x] = log_now[y, x]  # pixel's reference resets after firing

        for x, y in zip(neg_x, neg_y):
            events.append((x, y, t, -1))
            log_ref[y, x] = log_now[y, x]

    return events


# ----------------------------------------------------------------------
# 3. Visualize: accumulate all events into a single "event frame"
# ----------------------------------------------------------------------
def events_to_image(events, width: int, height: int) -> np.ndarray:
    """Standard event-camera visualization: blue = positive, red = negative."""
    img = np.full((height, width, 3), 255, dtype=np.uint8)  # white background
    for x, y, t, p in events:
        if p > 0:
            img[y, x] = (255, 0, 0)     # blue channel in BGR-style
        else:
            img[y, x] = (0, 0, 255)     # red channel
    return img


def main():
    frames = [render_frame(t) for t in range(N_FRAMES)]
    events = simulate_events(frames, CONTRAST_THRESHOLD)
    print(f"Simulated {len(events)} events over {N_FRAMES} frames "
          f"(threshold C={CONTRAST_THRESHOLD}).")

    event_img = events_to_image(events, WIDTH, HEIGHT)

    # Side-by-side comparison: last raw frame vs. accumulated events
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].imshow(frames[-1], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Conventional frame (last time step)")
    axes[0].axis("off")

    axes[1].imshow(event_img)
    axes[1].set_title(f"Accumulated DVS events ({len(events)} events)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("/home/claude/event_simulation_result.png", dpi=150)
    print("Saved visualization to event_simulation_result.png")

    # Also plot event rate over time — a common diagnostic in event-vision work
    times = np.array([e[2] for e in events])
    plt.figure(figsize=(6, 3))
    plt.hist(times, bins=N_FRAMES, color="black")
    plt.xlabel("Time step")
    plt.ylabel("Number of events")
    plt.title("Event rate over time")
    plt.tight_layout()
    plt.savefig("/home/claude/event_rate_over_time.png", dpi=150)
    print("Saved event-rate plot to event_rate_over_time.png")


if __name__ == "__main__":
    main()
