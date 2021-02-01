import os
import numpy as np
from PIL import Image, ImageDraw
import random
from tqdm import tqdm
import tensorflow as tf
import math


class CGHDataProvider:
	def __init__(self, num_data, slm_info, num_z_planes, z_sep, wavelength):
		self.path = os.getcwd()
		self.training_data_path = os.path.join(os.getcwd(), "DataProvider/training_data/")
		self.validation_data_path = os.path.join(os.getcwd(), "DataProvider/validation_data/")

		self.nT = num_data
		self.nV = int(num_data * 0.125)
		self.Mx = slm_info[0]
		self.My = slm_info[1]
		self.lp = slm_info[2]
		self.l0x = self.Mx * self.lp
		self.l0y = self.My * self.lp
		self.nz = num_z_planes
		self.dz = z_sep
		self.wl = wavelength * 10 ** -9
		self.batch_size = 32
		self._generate_training_data()
		self._calculate_phase_factors()

	# TRAINING AND VALIDATION DATA

	def _create_lines(self, N_images):
		line_images = np.zeros((N_images, self.Mx, self.My, self.nz))
		progress = tqdm(line_images)
		for image in progress:
			progress.set_description("Creating lines...")
			for plane in range(self.nz):
				im = Image.fromarray(image[:, :, plane])
				draw_im = ImageDraw.Draw(im)
				num_lines = random.randint(0, 10)
				for n_line in range(num_lines):
					start_x = random.randint(10, self.Mx - 10)
					start_y = random.randint(10, self.My - 10)
					stop_x = random.randint(10, self.Mx - 10)
					stop_y = random.randint(10, self.My - 10)
					draw_im.line((start_x, start_y, stop_x, stop_y), fill=random.randint(128, 256),
								 width=random.randint(1, 3))
				image[:, :, plane] = np.array(im)
		return line_images

	def _create_circles(self, N_images):
		circle_images = np.zeros((N_images, self.Mx, self.My, self.nz))
		progress = tqdm(circle_images)
		for image in progress:
			progress.set_description("Creating circles...")
			for plane in range(self.nz):
				im = Image.fromarray(image[:, :, plane])
				draw_im = ImageDraw.Draw(im)
				num_circles = random.randint(0, 10)
				for n_circle in range(num_circles):
					diameter = random.randint(2, int(self.Mx / 4))
					x_0 = random.randint(diameter, self.Mx - diameter)
					y_0 = random.randint(diameter, self.My - diameter)
					x_1 = x_0 + diameter
					y_1 = y_0 + diameter
					draw_im.ellipse([(x_0, y_0), (x_1, y_1)], outline=random.randint(0, 256),
									fill=random.randint(0, 256), width=random.randint(1, 3))
				image[:, :, plane] = np.array(im)
		return circle_images

	def _create_polygons(self, N_images):
		poly_images = np.zeros((N_images, self.Mx, self.My, self.nz))
		progress = tqdm(poly_images)
		for image in progress:
			progress.set_description("Creating polygons...")
			for plane in range(self.nz):
				im = Image.fromarray(image[:, :, plane])
				draw_im = ImageDraw.Draw(im)
				num_polys = random.randint(0, 10)
				for n_poly in range(num_polys):
					radius = random.randint(10, int(self.Mx / 4))
					x_0 = random.randint(radius, self.Mx - radius)
					y_0 = random.randint(radius, self.My - radius)
					n_sides = random.randint(3, 6)
					xs = [random.randint(x_0 - radius, x_0 + radius) for n in range(n_sides)]
					ys = [random.randint(y_0 - radius, y_0 + radius) for n in range(n_sides)]
					xy = [val for pair in zip(xs, ys) for val in pair]
					draw_im.polygon(xy, outline=random.randint(0, 256), fill=random.randint(0, 256))
				image[:, :, plane] = np.array(im)
		return poly_images

	def _generate_training_data(self, n_lines=0.5, n_circles=0.25, n_polys=0.25):
		assert n_lines + n_circles + n_polys == 1, "Training data split should add up to 1"
		nL = int(n_lines * self.nT)
		nC = int(n_circles * self.nT)
		nP = int(n_polys * self.nT)

		# Check whether training dataset exists already
		file_name = "TRAIN-Mx{}-My{}-nz{}-nT{}-nV{}-nL{}-nC{}-nP{}.tfrecords".format(self.Mx, self.My, self.nz, self.nT,
																					 self.nV, nL, nC, nP)
		if os.path.exists(os.path.join(self.training_data_path, file_name)):
			print("Chosen training data already exists. Continuing...")
		else:
			lines = self._create_lines(nL)
			circles = self._create_circles(nC)
			polys = self._create_polygons(nP)
			print("Combining training data shapes ...")
			training_data = np.concatenate((lines, circles, polys))
			print("Saving training data ...")
			with tf.io.TFRecordWriter(os.path.join(self.training_data_path, file_name)) as writer:
				progress = tqdm(training_data)
				for training_image in progress:
					progress.set_description("Writing training data to tfrecords ...")
					image_bytes = training_image.tostring()

					f = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))

					feature = {'image': f}

					features = tf.train.Features(feature=feature)
					example = tf.train.Example(features=features)
					example_to_string = example.SerializeToString()

					writer.write(example_to_string)

			print("Finished generating training data")
		self._generate_validation_data(n_lines, n_circles, n_polys)
		self.train_file_path = os.path.join(self.training_data_path, file_name)

	def _generate_validation_data(self, n_lines, n_circles, n_polys):
		assert n_lines + n_circles + n_polys == 1, "Training data split should add up to 1"
		nL = int(n_lines * self.nV)
		nC = int(n_circles * self.nV)
		nP = int(n_polys * self.nV)

		# Check whether validation dataset exists already
		file_name = "VAL-Mx{}-My{}-nz{}-nT{}-nV{}-nL{}-nC{}-nP{}.tfrecords".format(self.Mx, self.My, self.nz, self.nT,
																				   self.nV, nL, nC, nP)
		if os.path.exists(os.path.join(self.validation_data_path, file_name)):
			print("Chosen validation data already exists. Continuing...")
		else:
			lines = self._create_lines(nL)
			circles = self._create_circles(nC)
			polys = self._create_polygons(nP)
			print("Combining validation data shapes ...")
			validation_data = np.concatenate((lines, circles, polys))
			print("Saving validation data ...")
			with tf.io.TFRecordWriter(os.path.join(self.validation_data_path, file_name)) as writer:
				progress = tqdm(validation_data)
				for val_image in progress:
					progress.set_description("Writing validation data to tfrecords ...")

					image_bytes = val_image.tostring()
					f = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))

					feature = {'image': f}

					features = tf.train.Features(feature=feature)
					example = tf.train.Example(features=features)
					example_to_string = example.SerializeToString()

					writer.write(example_to_string)
			print("Finished generating validation data")
		self.val_file_path = os.path.join(self.validation_data_path, file_name)

	# FOURIER OPTICS SPECIFIC FUNCTIONS
	def _calculate_phase_factors(self):
		fx = np.linspace(-self.Mx / 2 + 1, self.Mx / 2, self.Mx) * 1 / (self.lp * self.Mx)
		fy = np.linspace(-self.My / 2 + 1, self.My / 2, self.My) * 1 / (self.lp * self.My)
		Fx, Fy = np.meshgrid(fx, fy)

		center = self.nz // 2
		phase_factors = []

		for n in range(self.nz):
			zn = n - center
			p = np.exp(-1j * math.pi * self.wl * (zn * self.dz) * (Fx ** 2 + Fy ** 2))
			phase_factors.append(p.astype(np.complex64))
		self.phase_factors = phase_factors

	def get_params(self, bs, ds, gs, ws):
		return {"Mx": self.Mx, "My": self.My, "nz": self.nz, "batch_size": self.batch_size, "bs": bs, "ds": ds, "gs": gs, "ws": ws}

	# TODO: Get training data https://keras.io/examples/keras_recipes/tfrecord/
	def get_training_data(self):
		return self.train_file_path, self.val_file_path
