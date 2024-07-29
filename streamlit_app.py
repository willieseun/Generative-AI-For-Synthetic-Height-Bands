import os
import sys
import torch
import shutil
import pickle
import rasterio
import tempfile
#import pgmagick
import importlib
import numpy as np
import pyspatialml
from glob import glob
import streamlit as st
from pyspatialml import Raster
from pyspatialml.datasets import nc
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gen_cnn_dem import CNN, resume

batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
print(torch.backends.cudnn.version())
model = CNN()
model.to(device)
resume(model, "gen_cnn_trial.pth")
geo_dict = {
	'driver': 'GTiff',
	'dtype': 'float32',
	'nodata': float('nan'),
	'width': 73751,
	'height': 59563,
	'count': 8,
	'crs': 'EPSG:32631',
	'transform': (3.0, 0.0, 431166.0, 0.0, -3.0, 969000.0),
	'blockysize': 1,
	'tiled': False,
	'compress': 'deflate',
	'interleave': 'pixel'
}
target_meta = geo_dict
#target_stack = Raster("/home/willieseun/Desktop/PervasiveAI/Oyo_planet_Mosaic_all.tif")
#dat = rasterio.open("/home/willieseun/Desktop/PervasiveAI/Oyo_planet_Mosaic_all.tif")
#profile = dat.profile.copy()
#print(profile)
#target_meta = target_stack.meta
#target_meta = profile
#print(target_meta)


with open('target_scaler_height.pkl', 'rb') as f:
	target_scaler = pickle.load(f)
st.title("Generative AI For Synthetic Height Bands (Courtesy of Ryzen AI-powered PCs)")

uploaded_files = st.file_uploader("Upload Sentinel 2 bands (B02, B03, B04, B08)", type="tif", accept_multiple_files=True)
temp_file_path_lst = []
if uploaded_files:
	for uploaded_file in uploaded_files:
		# Save the uploaded file temporarily
		with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as temp_file:
			temp_file.write(uploaded_file.read())
			temp_file_path = temp_file.name
		temp_file_path_lst.append(temp_file_path)

		st.success(f"File {uploaded_file.name} uploaded successfully!\nWait for processing")
	stack_obj = Raster(temp_file_path_lst)
	try:
		new_raster_file_path = 'New_raster.tif'
		result = stack_obj.torch_regression_dpl_output_scaler(estimator=model, shape=(2,5), scaler=target_scaler, target_meta=None, no_data=np.inf, dtype='float32', progress=True)
		result.write(new_raster_file_path)

		# Provide download link for the processed file
		with open(new_raster_file_path, "rb") as file:
			st.download_button(
				label=f"Download Upscaled Raster",
				data=file,
				file_name=f"Processed_{uploaded_file.name}",
				mime="image/tiff"
			)
	#if os.path.isfile(new_raster_file_path):
		#os.remove(new_raster_file_path)
		#print(f"File {new_raster_file_path} has been deleted.")
	#else:
	#	print(f"The file {new_raster_file_path} does not exist.")
	except:
		st.error("Error: Please upload the specified bands above and try again")

	# Clean up the temporary files
	#os.remove(temp_file_path)
