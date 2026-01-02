import cv2
import numpy as np
import argparse
import time
import imutils
from imutils.video import VideoStream

color_green = (0, 255, 0, 255)
box_contour_thickness = 2
max_objects = 5
current_frame = 0

def point_near_box(point, box, proximity = 100):
	point_x, point_y = point
	x, y, h, w = box
	return point_x > x - proximity and point_x < x + w + proximity and point_y > y - proximity and point_y < y + h + proximity  

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
	help="path to input video file")

args = vars(ap.parse_args())

# Assume stable camera, so background substractor will provide reliable result
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
multiTracker = cv2.legacy.MultiTracker_create()

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

	success, boxes = multiTracker.update(frame)
	for box in boxes:
		x, y, w, h = box
		cv2.rectangle(frame, (int(x), int(y)), (int(x) + int(w), int(y) + int(h)), color_green, box_contour_thickness)

	# Limit detection procedure to every 10th frame
	if current_frame % 10 == 0:
		mask = object_detector.apply(frame)
		_, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
		contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		for contour in contours:
			# Calculate area and skip small elements (most likely noise)
			area = cv2.contourArea(contour)
			if area > 200:
				x, y, w, h = cv2.boundingRect(contour)
				bb = (x, y, w, h)
				for box in boxes:
					if point_near_box((x, y), box):
						break
				else:
					# Add new tracker only when object detected far enough from currently tracked
					multiTracker.add(cv2.legacy.TrackerKCF_create(), frame, bb)
		
	current_frame += 1

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