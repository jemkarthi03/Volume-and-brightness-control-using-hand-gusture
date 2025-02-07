import cv2
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbc

from handLmModel import HandDetector

# Initialize Video Capture
vidObj = cv2.VideoCapture(0)
vidObj.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vidObj.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Initialize Hand Detector
handlmsObj = HandDetector(detectionCon=0.7)

# Initialize Audio Control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Brightness Control Limits
minBrightness = 0
maxBrightness = 100

# Store previous volume level to smooth transitions
prev_volume = None


def setVolume(dist):
    global prev_volume
    vol = np.interp(dist, [30, 200], [0.0, 1.0])  # Convert distance to volume scale (0.0 to 1.0)
    vol = round(vol, 2)  # Round to avoid excessive small changes

    # Only update volume if the change is significant
    if prev_volume is None or abs(prev_volume - vol) > 0.05:
        volume.SetMasterVolumeLevelScalar(vol, None)  # Set volume in percentage scale
        prev_volume = vol
        print(f"🔊 Volume Set To: {int(vol * 100)}%")  # Debug print


def setBrightness(dist):
    brightness = np.interp(dist, [30, 220], [minBrightness, maxBrightness])
    brightness = int(brightness)  # Convert to integer
    sbc.set_brightness(brightness)
    print(f"💡 Brightness Set To: {brightness}%")  # Debug print


while True:
    success, frame = vidObj.read()
    if not success:
        print("Failed to capture frame")
        break

    frame = cv2.flip(frame, 1)
    frame = handlmsObj.findHands(frame)
    lndmrks = handlmsObj.findPosition(frame, draw=False)

    if lndmrks and len(lndmrks) > 1:
        try:
            handType = lndmrks[0]

            if len(lndmrks) > 1 and len(lndmrks[1]) > 8:
                # Thumb (id=4) and Index Finger Tip (id=8) Coordinates
                x1, y1 = lndmrks[1][4][1], lndmrks[1][4][2]
                x2, y2 = lndmrks[1][8][1], lndmrks[1][8][2]

                # Calculate Distance
                dist = math.hypot(x2 - x1, y2 - y1)

                if handType == 'Left':
                    setBrightness(dist)  # Adjust brightness
                elif handType == 'Right':
                    setVolume(dist)  # Adjust volume

        except IndexError as e:
            print(f"Index Error: {e}")
        except Exception as e:
            print(f"Unexpected Error: {e}")

    cv2.imshow("Hand Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vidObj.release()
cv2.destroyAllWindows()
