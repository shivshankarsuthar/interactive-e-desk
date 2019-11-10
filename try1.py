from cv2 import *
import numpy as np

def nothing(x):
	pass
cap = VideoCapture(0)


focalLength = 650
actual_width = 6.5

namedWindow("Tracking")
namedWindow("Write")
cv2.createTrackbar("LH", "Tracking", 96, 255, nothing)
cv2.createTrackbar("LS", "Tracking", 68, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 181, 255, nothing)
cv2.createTrackbar("US", "Tracking", 192, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)
cv2.createTrackbar("value","Tracking",focalLength,5000,nothing)


def distance(pixel_distance,focal):
	dist = actual_width * focal / pixel_distance
	return dist


canvas = np.zeros([650,600,3],np.uint8)
x_old = 250
y_old = 250
is_drawing = False
points = []

while(cap.isOpened()):
	ret,frame = cap.read()
	frame = cv2.flip(frame,1)
	hsv = cvtColor(frame,COLOR_BGR2HSV)

	l_h = cv2.getTrackbarPos("LH", "Tracking")
	l_s = cv2.getTrackbarPos("LS", "Tracking")
	l_v = cv2.getTrackbarPos("LV", "Tracking")

	u_h = cv2.getTrackbarPos("UH", "Tracking")
	u_s = cv2.getTrackbarPos("US", "Tracking")
	u_v = cv2.getTrackbarPos("UV", "Tracking")

	value = getTrackbarPos("value","Tracking")

	l_b = np.array([l_h, l_s, l_v])
	u_b = np.array([u_h, u_s, u_v])

	mask = cv2.inRange(hsv, l_b, u_b)
	res = cv2.bitwise_and(frame, frame, mask=mask)
	imgray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(imgray, (5, 5), 0)
	edged = cv2.Canny(gray, 35, 125)
	_,contours,_ = findContours(edged,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE)
	largest_area = 0
	largest_contour_index = 0
	for i in range(len(contours)):
		peri = cv2.arcLength(contours[i], True)
		approx = cv2.approxPolyDP(contours[i], 0.04 * peri, True)
		if(len(approx) == 4 ):
			area = contourArea(contours[i])
			if area > largest_area:
				largest_area = area
				largest_contour_index = i
				#bounding_rect = bondingRect(contours[i])
		
	#x,y,w,h = cv2.boundingRect(contours[largest_contour_index])
	#cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
	#print(x,y,w,h)
#		rect = minAreaRect(cnt)
	#	box = boxPoints(rect)
		#box = np.int0(box)
	rect = cv2.minAreaRect(contours[largest_contour_index])
	box = cv2.boxPoints(rect)
	box = np.int0(box)
#	x_old = rect[0][0] - rect[1][0]/2
#	y_old = rect[1][0]
	print("%1.f , %1.f"  % (rect[0][0]-rect[1][0]/2,rect[1][0]))
	cv2.drawContours(frame,[box],0,(0,0,255),2)
	dist = distance(rect[1][0],value)
	cv2.putText(frame, 'Distance: %.2f cm' % dist, (10,30), FONT_HERSHEY_SIMPLEX ,
                   1, (0,255,0), 2, cv2.LINE_AA)
	#drawContours(frame,contours,largest_contour_index,(0,255,0),2)
	#print(largest_area)
#	drawContours(frame,contours,cnt,(0,0,255),2)
#	drawContours(frame,contours,-1,(0,255,0),2)
	#_,thresh = threshold(imgray,127,255,0)
#	imshow("mask",edged)
#	imshow("frame",frame)
#	imshow('canvas',canvas)
	k = waitKey(1)


	if k == ord('c'):
		canvas[:,:] = 255
		is_drawing = False
		points.clear()

	if k == ord('d'):
		is_drawing = True
		x_old = int(rect[0][0]-rect[1][0]/2)
		y_old = int(rect[1][0])


	if k == 27:
		break	


	if is_drawing:
		if len(points) > 100:
			points.pop(0)
		x,y = int(rect[0][0]-rect[1][0]/2),int(rect[1][0])
		points.append((x,y))
		
		for i in range(len(points)-1):
			if points[i+1] > (points[i][0] + 10,points[i][1]+10)  or points[i+1] < (points[i][0] - 10,points[i][1] + 10):
				continue
			cv2.line(canvas, points[i],points[i+1] , (0,255,0), 2)
			cv2.line(frame, points[i], points[i+1], (0,0,0), 2)

	imshow('canvas',canvas)
	imshow('frame',frame)
	imshow('mask',mask)
cap.release()
destroyAllWindows()
