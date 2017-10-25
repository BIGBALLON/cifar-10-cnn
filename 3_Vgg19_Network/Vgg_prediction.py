import keras
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input, decode_predictions
from PIL import Image
import numpy as np
import os.path

# Using ImageNet pre_trained weights to predict image's class(1000 class)
# ImageNet -- http://www.image-net.org/
# make sure your package pillow is the latest version 

model = VGG19(weights='imagenet') # load keras ImageNet pre_trained model

while True:
	img_path = input('Please input picture file to predict ( input Q to exit ):  ')
	if img_path == 'Q':
		break
	if not os.path.exists(img_path):
		print("file not exist!")
		continue
	try:
		img = Image.open(img_path)
		ori_w,ori_h = img.size
		new_w = 224.0;
		new_h = 224.0;
		if ori_w > ori_h:
			bs = 224.0 / ori_h;
			new_w = ori_w * bs
			weight = int(new_w)
			height = int(new_h)
			img = img.resize( (weight, height), Image.BILINEAR )
			region = ( weight / 2 - 112, 0, weight / 2 + 112, height)
			img = img.crop( region )
		else:
			bs = 224.0 / ori_w;
			new_h = ori_h * bs
			weight = int(new_w)
			height = int(new_h)
			img = img.resize( (weight, height), Image.BILINEAR )
			region = ( 0, height / 2 - 112 , weight, height / 2 + 112  )
			img = img.crop( region )
		x = np.asarray( img, dtype = 'float32' )
		x[:, :, 0] = x[:, :, 0] - 123.680
		x[:, :, 1] = x[:, :, 1] - 116.779
		x[:, :, 2] = x[:, :, 2] - 103.939
		x = np.expand_dims(x, axis=0)
		results = model.predict(x)
		print('Predicted:', decode_predictions(results, top=5)[0])
	except Exception as e:
		pass


	