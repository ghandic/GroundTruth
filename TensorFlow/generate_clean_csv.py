"""
Author:
	Andy Challis
Usage:
	python generate_clean_csv.py --csv_input=data/bokeh_result.csv --img_dir=../Images/ --output_path=data/ --train_percent=0.75

TODO:
	produce the pbtxt for single class and multiple classes
"""


import tensorflow as tf
from PIL import Image
import pandas as pd
import numpy as np

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('img_dir', '', 'Directory to where the images are stored')
flags.DEFINE_string('output_path', '', 'Path to output train and test CSVs')
flags.DEFINE_float('train_percent', 0.8, 'Percent of input dataset to be used for training')
FLAGS = flags.FLAGS

def clean_and_convert_csv(csv_input, img_dir):

	"""

	Converts the data we made using the Bokeh ground truthing tool into
	a usable dataset for TensorFlow

	csv_input, str - The path to the csv output from the Bokeh tool
	img_dir, str - The path to where the images are stored

	"""
	data_in = pd.read_csv(csv_input)
	data_out = []

	for image in data_in.index:
	    
	    in_img, in_x, in_y, in_w, in_h = data_in.iloc[image]
	    
	    
	    im = Image.open(img_dir+ in_img.split('/').pop())
	    im_w, im_h = im.size[0], im.size[1]
	    
	    out = {'filename': in_img.split('/').pop(),
	           'width': im_w,
	           'height': im_h,
	           'class': 'cat',
	           'xmin': np.floor((in_x - in_w/2)*im_w),
	           'ymin': np.floor((in_y - in_h/2)*im_h), 
	           'xmax': np.floor((in_x + in_w/2)*im_w),
	           'ymax': np.floor((in_y + in_h/2)*im_h)}
	    
	    data_out.append(out)
	
	return pd.DataFrame(data_out)


def split_dataset(data, split):

	"""

	A simple function to split the dataset given a percentage for the train size

	data, pd dataframe - The dataset to split
	split, float - The percentage of training samples, 
					eg 0.8 = 20% reserved for testing

	"""

	msk = np.random.rand(len(data)) < split
	train = data[msk]
	test = data[~msk]

	return train, test

def main(_):

	data_out = clean_and_convert_csv(FLAGS.csv_input, FLAGS.img_dir)
	train_set, test_set = split_dataset(data_out, FLAGS.train_percent)

	data_out.to_csv(FLAGS.output_path + 'full_records.csv', index=False,
               		 columns=['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
	print('Successfully created the clean dataset: {}'.format(FLAGS.output_path + 'full_records.csv'))

	train_set.to_csv(FLAGS.output_path + 'train_records.csv', index=False,
               		 columns=['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
	print('Successfully created the clean training set: {}'.format(FLAGS.output_path + 'train_records.csv'))

	test_set.to_csv(FLAGS.output_path + 'test_records.csv', index=False,
               		 columns=['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
	print('Successfully created the clean test set: {}'.format(FLAGS.output_path + 'test_records.csv'))

if __name__ == '__main__':
    tf.app.run()
