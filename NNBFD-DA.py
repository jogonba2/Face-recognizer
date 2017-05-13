#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Ejs:
# export PYTHONHASHSEED=1337
# [Eval]  > python NNBFD.py evaluate --faces_path ./dfFaces_24x24_norm --no_faces_path NotFaces_24x24_norm --pos_ini 0 --size 16099 --crop_w 24 --crop_h 24 --p_dev 0.2 --epochs 20
# [Train] > python NNBFD.py train --faces_path ./dfFaces_24x24_norm --no_faces_path NotFaces_24x24_norm --pos_ini 0 --size 16099 --model_name nn_model_da.h5 --epochs 100 --crop_w 24 --crop_h 24 --verbose 1
# [Test]  > python NNBFD.py test --test_file ./test3.jpg --model_name nn_model_da.h5 --crop_w 24 --crop_h 24 --stride_w 24 --stride_h 24 --subsampling 0.90 --verbose 1

from PIL import Image
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense
from keras.utils.np_utils import to_categorical
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from math import floor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
np.random.seed(1337)
import cv2

def plot(img, faces_xy):
	fig,ax = plt.subplots(1)
	ax.imshow(img)
	for i in range(len(faces_xy)):	
		width, height = faces_xy[i][3]-faces_xy[i][1], faces_xy[i][2]-faces_xy[i][0]
		rect = patches.Rectangle((faces_xy[i][0], faces_xy[i][1]),width, height,linewidth=1,edgecolor='r',facecolor='none')
		ax.add_patch(rect)
	plt.show()

def tag_faces(img_file, nn_model, CROP_IM_W, CROP_IM_H, STRIDE_W, STRIDE_H, SUBSAMPLING, VERBOSE):
	im = Image.open(img_file)
	im_or = im.copy()
	w, h = im.size
	faces_xy = []
	i = 0
	while im.width>=CROP_IM_W and im.height>=CROP_IM_H: # Subsampling
		im.thumbnail((w, h), Image.ANTIALIAS)
		# Cropping #
		regions   = []
		positions = []
		for j in range(0, w, STRIDE_W): #, CROP_IM_W):
			for p in range(0, h, STRIDE_H):# CROP_IM_H):
				if VERBOSE: print((j, p, CROP_IM_W, CROP_IM_H))
				regions.append(im.crop((j, p, j+CROP_IM_W, p+CROP_IM_H)))
				regions[-1] = regions[-1].convert("L") # Grayscale #
				positions.append((j, p, j+CROP_IM_W, p+CROP_IM_H))
		# Face extraction #
		for j in range(len(regions)):
			region = np.array((regions[j].getdata()))
			region = (region - np.mean(region)) / np.std(region)
			res = np.argmax(nn_model.predict(np.array([region])))
			if res==1:
				if VERBOSE: print("FIND", positions[j])
				faces_xy.append((positions[j][0]/(SUBSAMPLING**i), positions[j][1]/(SUBSAMPLING**i),
								 positions[j][2]/(SUBSAMPLING**i), positions[j][3]/(SUBSAMPLING**i)))
				if VERBOSE: print("ORIG", faces_xy[-1])
		w = int(w*SUBSAMPLING)
		h = int(h*SUBSAMPLING)
		i += 1
	plot(im_or, faces_xy)	

def build_nn(CROP_IM_W, CROP_IM_H):
	nn_model = Sequential()
	nn_model.add(BatchNormalization(input_shape=(CROP_IM_W*CROP_IM_H,)))
	nn_model.add(Dense(128, activation="relu"))
	nn_model.add(BatchNormalization())
	nn_model.add(Dense(64, activation="relu"))
	nn_model.add(BatchNormalization())
	nn_model.add(Dense(32, activation="relu"))
	nn_model.add(BatchNormalization())
	nn_model.add(Dense(2, activation="softmax"))
	nn_model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=["accuracy"])
	return nn_model

def load_corpus(file_faces, file_no_faces, pos_ini, size):
	faces      = np.genfromtxt(file_faces, delimiter=" ")
	y_faces    = [1 for i in range(faces.shape[0])]
	no_faces   = np.genfromtxt(file_no_faces, delimiter=" ")
	y_no_faces = [0 for i in range(no_faces.shape[0])]
	X = np.concatenate((faces, no_faces), axis=0)
	Y = np.concatenate((y_faces, y_no_faces), axis=0)
	X = X[pos_ini:size]
	Y = Y[pos_ini:size]
	return X, Y

def train_model(nn_model, faces_path, no_faces_path, pos_ini, size, epochs, model_name, VERBOSE=True, batch_size=32):
	X, Y = load_corpus(faces_path, no_faces_path, pos_ini, size)
	Y    = to_categorical(Y)
	datagen = ImageDataGenerator(featurewise_center=True, 
								 samplewise_center=False, 
								 featurewise_std_normalization=True, 
								 samplewise_std_normalization=False,
								 rotation_range=20,
								 horizontal_flip=True)
	datagen.fit(X.reshape((X.shape[0], 1, 24, 24)))	 
	for e in range(epochs):
		batches = 0
		loss = 0
		for X_batch, Y_batch in datagen.flow(X.reshape((X.shape[0], 1, 24, 24)), Y, batch_size=batch_size):
			if batches >= floor(float(X.shape[0]) / batch_size)-1: break
			X_batch = X_batch.reshape((32, 576))
			loss += nn_model.train_on_batch(X_batch, Y_batch)[0]
			batches += 1
		avg_loss = loss / batches
		nn_model.fit(X, Y, batch_size=batch_size, nb_epoch=1)
		print("Epoch %d, loss %f\n" % (e, avg_loss))
	nn_model.save(model_name)
	return nn_model

def evaluate(X_train, Y_train, X_dev, Y_dev, nn_model, epochs):
	Y_train_cat = to_categorical(Y_train)
	Y_dev_cat   = to_categorical(Y_dev)
	nn_model.fit(X_train, Y_train_cat, nb_epoch=epochs, verbose=True)
	pred        = [np.argmax(p) for p in nn_model.predict(X_dev)]
	C = confusion_matrix(Y_dev, pred)
	TN, FN, TP, FP = C[0][0], C[1][0], C[1][1], C[0][1]
	print("\n\n\t--- Confusion matrix ---\n")
	print("\t ¬Cara\t Cara\t\n")
	print("¬Cara\t", TN, "\t", FP, "\n")
	print(" Cara\t", FN, "\t", TP, "\n")
	print("\n* TN=%d\tFN=%d\tTP=%d\tFP=%d" % (TN, FN, TP, FP))
	
def shuffle_unison(x, y):
	p = np.random.permutation(len(x))
	return x[p], y[p]
	
def main(args):
	nn_model = build_nn(args.crop_w, args.crop_h)
	if args.method=="train":
		nn_model = train_model(nn_model, args.faces_path, 
							   args.no_faces_path, args.pos_ini, 
							   args.size, args.epochs, args.model_name, bool(args.verbose))
	elif args.method=="test":
		nn_model = load_model(args.model_name)
		tag_faces(args.test_file, nn_model, args.crop_w, args.crop_h, args.stride_w, args.stride_h, args.subsampling, bool(args.verbose))
	
	elif args.method=="evaluate":	
		X, Y = load_corpus(args.faces_path, args.no_faces_path, args.pos_ini, args.size)
		X, Y = shuffle_unison(X, Y)
		X_train, Y_train = X[0:int((1-args.p_dev)*len(X))], Y[0:int((1-args.p_dev)*len(Y))]
		X_dev, Y_dev = X[int((1-args.p_dev)*len(X)):], Y[int((1-args.p_dev)*len(X)):]
		evaluate(X_train, Y_train, X_dev, Y_dev, nn_model, args.epochs)
		
	else: exit()   

if __name__ == "__main__": 
	import argparse
	parser = argparse.ArgumentParser(description='Neural network based face detection')
	parser.add_argument('method', action="store", type=str)
	parser.add_argument('--faces_path', action="store", type=str)
	parser.add_argument('--no_faces_path', action="store", type=str)
	parser.add_argument('--pos_ini', action="store", type=int)
	parser.add_argument('--size', action="store", type=int)
	parser.add_argument('--epochs', action="store", type=int)	
	parser.add_argument('--model_name', action="store", type=str)
	parser.add_argument('--test_file', action="store", type=str)
	parser.add_argument('--crop_w', action="store", type=int)
	parser.add_argument('--crop_h', action="store", type=int)
	parser.add_argument('--stride_w', action="store", type=int)
	parser.add_argument('--stride_h', action="store", type=int)
	parser.add_argument('--p_dev', action="store", type=float)
	parser.add_argument('--subsampling', action="store", type=float)
	parser.add_argument('--verbose', action="store", type=int)
	main(parser.parse_args())
