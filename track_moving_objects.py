import cv2
import numpy as np
import argparse
import time
import imutils
from imutils.video import FPS
from imutils.video import VideoStream
import countours_merger
from countours_merger import agglomerative_cluster

color_green = (0, 255, 0)
box_contour_thickness = 3

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
	help="path to input video file")

args = vars(ap.parse_args())

# Assume stable camera, so background substractor will provide reliable result
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
# More reliable tracking algo for purposes of this task
tracker = cv2.TrackerCSRT_create()

if not args.get("video", False):
	# open first webcam by default
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(1.0) #camera "warmup"
    
else:
	vs = cv2.VideoCapture(args["video"])

while True:
	frame = vs.read()
	if args.get("video", False):
		frame = frame[1]
	else:
		frame = cv2.flip(frame, 1)
	# check to see if we have reached the end of the stream (in case it is a video)
	if frame is None:
		break
	# resize the frame (so it can be processed faster) and grab the frame dimensions
	frame = imutils.resize(frame, width=500)
	(H, W) = frame.shape[:2]

	mask = object_detector.apply(frame)
	_, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
	contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# contours = agglomerative_cluster(contours) # Need adaptation for newer versios of CV2

	detections = []

	for contour in contours:
        # Calculate area and skip small elements (most likely noise)
		area = cv2.contourArea(contour)
		if area > 100:
			# cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
			x, y, w, h = cv2.boundingRect(contour)
			# cv2.putText(frame, str(id), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, color_green, int(box_contour_thickness))
			cv2.rectangle(frame, (x, y), (x + w, y + h), color_green, box_contour_thickness)
	
				
	cv2.imshow("Frame", frame)
	# cv2.imshow("Mask", mask)

	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

if not args.get("video", False):
	vs.stop()
else:
	vs.release() # Release the file pointer
time.sleep(1.0)  # Allow stream to end gracefully
cv2.destroyAllWindows()