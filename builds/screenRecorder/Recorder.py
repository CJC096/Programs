import cv2
import numpy as np
import mss
import time

# Screen capture setup
resolution = (1920, 1080)  # Adjust to your actual screen size
fps = 60.0
filename = "Recording.avi"
codec = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(filename, codec, fps, resolution)

# Create resizable preview window
cv2.namedWindow('Live Preview', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Live Preview', 640, 360)

# Allow preview window to appear before recording starts
time.sleep(1)

with mss.mss() as sct:
    monitor = sct.monitors[1]  # Primary monitor

    while True:
        sct_img = sct.grab(monitor)
        frame = np.array(sct_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        out.write(frame)
        cv2.imshow('Live Preview', frame)

        # Stop if user presses 'q' or closes the window
        if cv2.waitKey(1) == ord('q'):
            break
        if cv2.getWindowProperty('Live Preview', cv2.WND_PROP_VISIBLE) < 1:
            break

out.release()
cv2.destroyAllWindows()
