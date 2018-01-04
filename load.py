import csv
import cv2
import numpy as np
import math
import time
import random
import pickle

save_data = False
use_pre_split_data = True

if not use_pre_split_data:
	# If you want to get a new split of training and validation data
	lines = []
	images = []
	measurements = []
	training_percent = 0.8
	with open('C:/Users/markr/Desktop/windows-sim/windows_sim/out_mark/driving_log.csv') as csvfile:
	#with open('D:/CarND-Behavioral-Cloning-P3/data/data/driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)

	random.shuffle(lines)
	num_lines = len(lines)
	num_train = math.floor(training_percent*(num_lines))
	lines_train = lines[0:num_train]
	lines_valid = lines[num_train + 1:-1]

	def gen_data(lines, batch_size, num_batches):

		images_per_line = 3

		while 1:
			# Iterate through required batches. Yield a result for each batch
			for num in range(num_batches):
				final_images = np.ndarray(shape=(batch_size, 160, 320, 3), dtype=float)
				final_angles = np.ndarray(shape=(batch_size), dtype=float)
				lines_mod = lines[num*math.floor(batch_size/images_per_line) : (num + 1)*math.floor(batch_size/images_per_line)]
				for o, line in enumerate(lines_mod):
					# Iterate through the 3 camera images
					for i in range(3):
						# Get the three image paths (centre, left, right)
						source_path = line[i]
						filename = source_path.split('\\')[-1]
						current_path = 'C:/Users/markr/Desktop/windows-sim/windows_sim/out_mark/IMG/' + filename
						#current_path = 'D:/CarND-Behavioral-Cloning-P3/data/data/IMG/' + filename
						image = cv2.imread(current_path)
						if i == 0:
							# No correction factor required for centre camera
							measurement = float(line[3])
						elif i == 1:
							# Apply positive steering angle correction factor for left camera
							measurement = float(line[3]) + 0.2
						elif i == 2:
							# Apply negative steering angle correction factor for right camera
							measurement = float(line[3]) - 0.2

						flip = cv2.flip(image, 1)

						final_images[o*images_per_line + i] = image
						#final_images[o*6 + i + 3] = flip
						final_angles[o*images_per_line + i] = measurement
						#final_angles[o*6 + i + 3] = measurement*-1.0
				
				yield (final_images, final_angles)

	if save_data:
		with open('training_validation_datasets.pkl', 'wb') as f:
		    pickle.dump([lines_train, lines_valid], f)


	train_data = gen_data(lines_train, batch_size_train, num_batch_train)
	valid_data = gen_data(lines_valid, batch_size_valid, num_batch_valid)

else:
	# If you want to use a previously saved data split
	[lines_train, lines_valid] = pickle.load( open('training_validation_datasets.pkl', "rb" ) )
	batch_size_train = 18
	num_batch_train = math.floor(len(lines_train)/batch_size_train)
	batch_size_valid = 18
	num_batch_valid = math.floor(len(lines_valid)/batch_size_valid)

print('Number of training labels: {}'.format(len(lines_train)))
print('Number of validation labels: {}'.format(len(lines_valid)))