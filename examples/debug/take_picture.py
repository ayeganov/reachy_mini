"""Demonstrate how to make Reachy Mini look at a point in an image.

When you click on the image, Reachy Mini will look at the point you clicked on.
It uses OpenCV to capture video from a camera and display it, and Reachy Mini's
look_at_image method to make the robot look at the specified point.

Note: The daemon must be running before executing this script.
"""

import time

import cv2

from reachy_mini import ReachyMini


def main() -> None:
    """Get a frame and take a picture."""
    with ReachyMini() as reachy_mini:
        frame = reachy_mini.media.get_frame()
        start_time = time.time()
        while frame is None:
            if time.time() - start_time > 20:
                print("Timeout: Failed to grab frame within 20 seconds.")
                exit(1)
            print("Failed to grab frame. Retrying...")
            frame = reachy_mini.media.get_frame()
            time.sleep(1)

        cv2.imwrite("reachy_mini_picture.jpg", frame)
        print("Saved frame as reachy_mini_picture.jpg")


if __name__ == "__main__":
    main()
