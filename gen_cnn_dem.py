import os
import sys
import torch
import shutil
import pickle
import geopandas
import numpy as np
import torchvision
import pandas as pd
import torch.nn as nn
from glob import glob
from tqdm import tqdm
import rasterio as rio
from pickle import dump
from pickle import load
from copy import deepcopy
import torch.optim as optim
from pandas import read_csv
from pyspatialml import Raster
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pyspatialml.datasets import nc
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score as sk_r2_score
from sklearn.model_selection import train_test_split
from transformers import get_cosine_schedule_with_warmup
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, Normalizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
	#new_alloc = torch.cuda.memory.CUDAPluggableAllocator('/home/willieseun/Downloads/torch-apu-helper-main/alloc.so','gtt_alloc','gtt_free');
	#torch.cuda.memory.change_current_allocator(new_alloc)
	#print("Custom allocator set.")
	print(f"Allocated memory: {torch.cuda.memory_allocated()} bytes")
	print(f"Cached memory: {torch.cuda.memory_reserved()} bytes")


	#device = "cpu"
	print(torch.backends.cudnn.version())
	#predictors = glob('/home/willieseun/Desktop/PervasiveAI/sar_sentinel2_oyo_combined.tif')
	#stack_obj = Raster(predictors)
	#print(stack_obj.names)
	#single_layer = stack_obj.iloc[0]
	#print(single_layer)
	#stack_lst = []
	#sar_idx = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
	#sen_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	#for band_idx in range(20):
	#	if band_idx not in sar_idx:
	#		stack_lst.append(stack_obj.iloc[band_idx])
	#single_layer_lst = [single_layer, stack_obj.iloc[11], stack_obj.iloc[12]]
	#raster_layer_lst = []
	#for name, layer in stack_obj:
	#	raster_layer_lst.append(layer)
	#stack_lst = raster_layer_lst + single_layer_lst
	#stack = Raster(stack_lst)
	#print(stack.names)

	#training_py = geopandas.read_file('training_oyo_20244_add.shp')

	#df_polygons = stack.extract_vector(training_py, progress=True)
	#df_polygons = pd.read_csv('Finetuned_Polygons.csv')
	df_polygons = pd.read_csv('/home/willieseun/Desktop/PervasiveAI/stack2/Polygons_Sep_MSRGB_banana.csv')
	df_polygons = df_polygons.dropna()
	#df_polygons[["RED_fill", "GREEN_fill", "BLUE_fill"]] = df_polygons[["RED", "GREEN", "BLUE"]]
	#print(df_polygons)
	for col in df_polygons.columns:
	    print(col)
	#class_mapping = {"Cassava": 1, "Maize": 2, "Forest": 4, "Built up": 3, "Grassland_Shrubland": 5, "Water body": 6}
	#df_polygons["Classvalue"] = df_polygons["Class"].replace(class_mapping)

	# replace NaNs in training data
	#df_polygons = df_polygons.replace(np.nan, 0)
	#df_polygons = df_polygons.merge(
	#    right=training_py.loc[:, ["Classname", "Classvalue", "TestData"]],
	#    left_on="geometry_idx",
	#    right_on="index",
	#    right_index=True
	#)
	#train_rows = df_polygons[df_polygons['TestData'] == 0]
	#test_rows = df_polygons[df_polygons['TestData'] == 1]
	#planet+sen only
	X = df_polygons[["MSSEP2020_NIR", "RGBSEP2020_BLUE", "RGBSEP2020_GREEN", "RGBSEP2020_RED"]].values
	y = df_polygons[["MSSEP2020_HEIGHT", "RGBSEP2020_HEIGHT"]].values
	X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	print(X_train.shape)
	X_train = np.reshape(X_train, (X_train.shape[0], 2, 2))
	X_test = np.reshape(X_test, (X_test.shape[0], 2, 2))
	target_scaler = MinMaxScaler(feature_range=(1e-5, 1))
	Y_test_originial = Y_test
	Y_train = target_scaler.fit_transform(Y_train)
	Y_test = target_scaler.transform(Y_test)
	with open('target_scaler_height.pkl', 'wb') as f:
		pickle.dump(target_scaler, f)

	X_train = torch.from_numpy(X_train)
	Y_train = torch.from_numpy(Y_train)
	X_test = torch.from_numpy(X_test)
	Y_test = torch.from_numpy(Y_test)
	#y = tf.keras.utils.to_categorical(y-1, num_classes=5)
	#X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X, y, test_size=0.9, random_state=7)
	#X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=7)
	X_train.to(device)
	X_test.to(device)
	Y_train.to(device) #Put Y_train to gpu
	Y_test.to(device) #Put Y_test to gpu
	print('Done Splitting')
	print(X_train.shape)
	print(Y_train.shape)
	# Define the CNN architecture
	#, activity_regularizer=l1(0.001), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), kernel_constraint=max_norm(3), bias_constraint=max_norm(3)

	train_dataset = torch.utils.data.TensorDataset(X_train, Y_train) #Convert tensors to a tensor dataset
	test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

def checkpoint(model, filename):
	torch.save(model.state_dict(), filename)
	
def resume(model, filename):
	model.load_state_dict(torch.load(filename, map_location=device))


class CNN(torch.nn.Module):
	def __init__(self):
		super(CNN, self).__init__()

		# Convolutional layers
		self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(128, 148, kernel_size=3, padding=1)
		self.flat = nn.Flatten()
		# Batch normalization layer
		self.bn = nn.BatchNorm2d(148)

		# Pooling layer
		self.pool = nn.MaxPool2d(2, 2)

		# Fully connected layers
		self.fc1 = nn.Linear(296, 148)
		self.fc2 = nn.Linear(148, 2)

	def forward(self, x):
		x = x.unsqueeze(1)
		x = x.float()
		x = x.to(device)
		x = self.conv1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.relu(x)
		x = self.bn(x)
		x = self.pool(x)

		# Flatten the tensor before passing it to the fully connected layers
		x = self.flat(x)

		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)

		return x


if __name__ == "__main__":
	model = CNN()
	model.to(device)
	optimizer = optim.Adam(model.parameters(), lr=0.0003)
	#optimizer = optim.AdamW(model.parameters())
	loss_fn = nn.SmoothL1Loss(reduction='mean').to(device)
	n_epochs = 50

	early_stop_thresh = 15
	best_loss = -np.inf
	best_epoch = -1

	# Calculate total training steps (adjust based on your training setup)
	num_training_steps = len(train_loader) * n_epochs

	# Adjust num_warmup_steps based on your needs (here, using 10% of total steps)
	num_warmup_steps = int(0.1 * num_training_steps)

	# Create learning rate scheduler
	#lr_scheduler = get_cosine_schedule_with_warmup(
	#    optimizer=optimizer,
	#    num_warmup_steps=num_warmup_steps,
	#    num_training_steps=num_training_steps,
	#    num_cycles=0.5  # Adjust for multiple cycles if needed
	#)

	y_pred_list = []
	targets_list = []
	for epoch in tqdm(range(n_epochs)):
		#break
		model.train()
		total_loss = 0.0
		optimizer_state_dict = optimizer.state_dict()

		# Iterate over training batches
		for inputs, targets in tqdm(train_loader, desc="Training"):
			inputs = inputs.float().to(device)
			targets = targets.to(device)
			y_pred = model(inputs)
			loss = loss_fn(y_pred, targets.to(device))
			optimizer.zero_grad()
			#print("train_y_pred", y_pred)
			#print("train_targets", targets)
			loss.backward()
			optimizer.step()
			total_loss += loss.item()
			inputs = inputs.to('cpu')
			targets = targets.to('cpu')
			del inputs, targets
			#lr_scheduler.step()
		
		#checkpoint_state(model, f"checkpoint_epoch_{epoch}.pth", training_state)
		# Calculate average training loss for the epoch
		average_train_loss = total_loss / len(train_loader)
		print(average_train_loss)

		# Validation loop
		model.eval()
		acc = 0
		count = 0
		torch.cuda.empty_cache()

		with torch.no_grad():
			# Iterate over validation batches
			total_val_loss = 0.0
			for inputs, targets in tqdm(test_loader, desc="Validation"):
				#targets = targets.unsqueeze(0)
				#inputs = inputs.unsqueeze(0)
				inputs = inputs.float().to(device)
				targets = targets.to(device)
				with torch.no_grad():
					y_pred = model(inputs)
					loss = loss_fn(y_pred, targets)
				total_val_loss += loss
				inputs = inputs.to('cpu')
				targets = targets.to('cpu')
				y_pred_r2s = y_pred.cpu().detach().numpy()
				r2valscore = sk_r2_score(y_pred_r2s, targets, multioutput='raw_values')
				del inputs, targets
			average_val_loss = total_val_loss / len(test_loader)
			print("Epoch %d: model loss %.4e" % (epoch, average_val_loss))
			print("Epoch %d: model r2score " % (epoch), r2valscore)
			r2valscore = torch.mean(torch.tensor(r2valscore)).numpy()
			if r2valscore > best_loss:
				best_loss = r2valscore
				best_epoch = epoch
				checkpoint(model, "gen_cnn_height_trial.pth")
			elif epoch - best_epoch > early_stop_thresh:
				print("Early stopped training at epoch %d, with best loss of %.4e" % (epoch, best_loss))
				break  # terminate the training loop
		print("Best Loss So Far: %.4e" % best_loss)

	resume(model, "gen_cnn_height_trial.pth")
	print(model)

	Test_dataloader = torch.utils.data.DataLoader(X_test, batch_size=16)  # Adjust batch size as needed

	all_predictions = []
	all_targets = []

	# Loop through batches for evaluation
	for batch_x in tqdm(Test_dataloader, desc="Final Testing"):
	    batch_x = batch_x.to(device)  # Move batch to device
	    batch_y_pred = model(batch_x)
	    batch_y_pred = batch_y_pred.cpu().detach().numpy()  # Transfer back to CPU

	    # Process and append predictions and targets
	    #batch_y_pred = batch_y_pred.reshape(-1, batch_y_pred.shape[-1])
	    all_predictions.extend(target_scaler.inverse_transform(batch_y_pred))


	# Calculate metrics after processing all batches
	mae = mean_absolute_error(Y_test_originial, all_predictions, multioutput='raw_values')
	mse = mean_squared_error(Y_test_originial, all_predictions, multioutput='raw_values')
	rmse = np.sqrt(mse)
	r2s = sk_r2_score(Y_test_originial, all_predictions, multioutput='raw_values')

	# Print the results
	print('Mean absolute error:', mae)
	print('Mean squared error:', mse)
	print('Root Mean squared error:', rmse)
	print('R squared error:', r2s)
	del df_polygons, X_train, X_test, Y_train, Y_test#, X_train1, X_test1, Y_train1, Y_test1

	#predict raster
	#print('Starting to Predict')
	#result = stack.torch_regression_dpl_output_scaler(estimator=model, shape=(2,5), scaler=target_scaler, no_data=0, dtype='int16', progress=True)
	#print('Displaying Predicted Map Soon')
	#result.write('Gen_Planet.tif')

	#result.plot()
	#plt.show()
