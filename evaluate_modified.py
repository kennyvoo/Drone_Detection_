from core import Core
import cv2
import argparse
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("-f","--file_path",required=True,help="the file path for the image/video from the current directory")
parser.add_argument("-v","--video",default=False, action='store_true',help="Whether the file is a video")
parser.add_argument("-z","--zoom",default=250, type=int,help="enlarge in %")
parser.add_argument("-s","--section",default=4, type=int,choices={1,4, 6},help="1,4 or 6 only for now")

args=parser.parse_args()
c = Core()
#Get the detection model
c.set_model(c.get_model())


file_name = c.current_path + args.file_path
section=args.section

#the increase percentage in size of the  original image
scale_percent =args.zoom


def split_to_sections(c,section,original_image,scale_percent):

	if (section==4):
		num=2 # if split into 4 section
	else: num=3 	#if split into 6 section

	height,width,_=original_image.shape
	scale=1
	min_row=0
	max_row=np.floor(height/2)
	min_column=0
	max_column=np.floor(width/num)

	drawing_image=original_image.copy()

	#calculate the increased size of the image
	resized_width = int(original_image[int(min_row):int(max_row),int(min_column):int(max_column)].shape[1] * scale_percent / 100)
	resized_height = int(original_image[int(min_row):int(max_row),int(min_column):int(max_column)].shape[0] * scale_percent / 100)
	dim = (resized_width, resized_height)

	for i in range(2):
		min_column=0
		max_column=int(width/num)
		for i in range(num):
			resized = cv2.resize(original_image[int(min_row):int(max_row),int(min_column):int(max_column)], dim, interpolation = cv2.INTER_AREA)
			boxes, scores, labels = c.predict_with_graph_loaded_model(resized, scale)
			detections = c.draw_boxes_in_image(drawing_image[int(min_row/scale):int(max_row/scale),int(min_column/scale):int(max_column/scale)], boxes/(scale_percent/100), scores)
			min_column=np.floor(width/num)
			max_column=np.floor(max_column*num)
		min_row=np.floor(height/2)
		max_row=np.floor(max_row*2)
	return drawing_image



if(args.video==False):
	image = c.load_image_by_path(file_name)
	original_image = c.get_drawing_image(image)
	#processed_image, scale = c.pre_process_image(image)
	processed_image=image.copy()
	scale=1
	height,width,_=processed_image.shape
	drawing_image=original_image.copy()
	#resized = cv2.resize(processed_image, dim, interpolation = cv2.INTER_AREA)
	boxes, scores, labels = c.predict_with_graph_loaded_model(drawing_image, scale)
	detections = c.draw_boxes_in_image(drawing_image, boxes, scores)
	print("Without splitting")
	c.visualize(drawing_image)

	print("splitted into "+str(section))
	if section!=1:   # without splitting to sections
	 # split into 4 or 6
		 drawing_image=split_to_sections(c,section,original_image,scale_percent)
	c.visualize(drawing_image)


else:
	# Added to perform detection on video 
	cap = cv2.VideoCapture(file_name)
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			print(" Exiting ...")
			break

		image = cv2.resize(frame, (800,600 )) 
		# This get_drawing_image actually just convert bgr to rgb
		original_image = c.get_drawing_image(image)
		drawing_image=original_image.copy
		if section==1:
			boxes, scores, labels = c.predict_with_graph_loaded_model(drawing_image, 1)
			detections = c.draw_boxes_in_image(drawing_image, boxes, scores)
			
		else:
			drawing_image=split_to_sections(c,section,original_image,scale_percent)

		cv2.imshow('Detections', drawing_image)
		if cv2.waitKey(1) == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()