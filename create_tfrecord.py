import os
import tensorflow as tf
from PIL import Image
from sys import argv

def generate(crop_path, record_path, output_name):
	path = crop_path
	if path[len(path) - 1] != '/':
		path += '/'

	counter = 0
	wc = 0
	rc_path = record_path + output_name
	record = rc_path + str(wc) + ".tfrecords"
	writer = tf.python_io.TFRecordWriter(record)

	for name in os.listdir(crop_path):
		img_path = path + name
		img = Image.open(img_path)

		key = name.split('_')
		key = int(key[len(key) - 1][0])

		img_raw = img.tobytes()
		example = tf.train.Example(features=tf.train.Features(feature={
			"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[key])),
			'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
		}))
		writer.write(example.SerializeToString())
		counter += 1
		if counter % 1000 == 0:
			print "finish writing: %d in %r" % (counter, writer)

		if counter % 500000 == 0:
			wc += 1
			writer.close()
			record = rc_path + str(wc) + ".tfrecords"
			writer = tf.python_io.TFRecordWriter(record)

	print "finish. Total: %d" % counter
	writer.close()

if __name__ == "__main__":
	generate(argv[1], argv[2], argv[3])