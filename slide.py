import openslide as osl
import xml.dom.minidom
import numpy as np
import os
from sys import argv

class OpenSlideReader(object):
	def __init__(self, img_path, mask_path, xml_path):
		tif = osl.OpenSlide(img_path)
		mask = osl.OpenSlide(mask_path)
		self.img = osl.OpenSlide(img_path)
		self.mask = osl.OpenSlide(mask_path)
		self.top_level = self.mask.level_count - 1

		self.img_path = img_path
		self.size = tif.level_dimensions

		self.pathological_region = self.pathological_reader(xml_path)
		self.normal_region = self.normal_reader(self.pathological_region)
		self.pathological_region_num = len(self.pathological_region)

		
		if self.img.dimensions != self.mask.dimensions:
			print "img not consistent with mask!"
			self.img = None
			self.mask = None
			return

		print "image read: %r, mask %r, size %r top level %r" % (img_path, mask_path, self.size[0], self.top_level)

	def rgb_img(self, level, left_top, size):
		return self.img.read_region(left_top, level, size)

	def rgb_mask(self, level, left_top, size):
		return self.mask.read_region(left_top, level, size)

	def img_size(self, level = 0):
		return self.size[level]

	def pathological_reader(self, xml_path):
		rxml = xml.dom.minidom.parse(xml_path)
		doc = rxml.documentElement

		annotations = doc.getElementsByTagName('Annotation')
		region_num = len(annotations)

		regions = []

		for a in annotations:
			try:
				data = a.getElementsByTagName('Coordinate')
				xs = np.array([d.getAttribute('X') for d in data]).astype('float')
				ys = np.array([d.getAttribute('Y') for d in data]).astype('float')
				regions.append((np.min(xs), np.min(ys), np.max(xs), np.max(ys)))

			except:
				print 'read xml fail'

		return regions

	def normal_reader(self, regions):
		new_regions = [] 
		for r in regions:
			min_x = r[0]
			min_y = r[1]
			max_x = r[2]
			max_y = r[3]
			dx = (max_x - min_x)
			dy = (max_y - min_y)
			nmin_x = max(min_x - dx,0)
			nmin_y = max(min_y - dy, 0)
			nmax_x = min(max_x + dx, self.size[0][0])
			nmax_y = min(max_y + dy, self.size[0][1])
			new_regions.append((nmin_x, nmin_y, nmax_x, nmax_y))

		return new_regions

	def crop_img(self, size, level, normal_num, pathological_num, target_path):
		finish = False
		current_normal = 0
		current_pathological = 0
	
		while not finish:
			randr = np.random.randint(self.pathological_region_num)
			#crop normal
			#sample coordinate
			region_xy = self.pathological_region[randr]

			min_x = region_xy[0]
			min_y = region_xy[1]
			max_x = region_xy[2]
			max_y = region_xy[3]

			# print region_xy

			randx = int(np.random.ranf(1) * (max_x - min_x) + min_x)
			randy = int(np.random.ranf(1) * (max_y - min_y) + min_y)
			if current_normal < normal_num:
				normal_mask = self.rgb_mask(level, (randx, randy), size)
				#covert to array
				ary = np.asarray(normal_mask)
				a = np.sum(ary, 2)
				#black (0, 0, 0, 255)
				#white (255, 255, 255, 255) 
				if np.max(a) == 255:
					current_normal = current_normal + 1
					
					normal_img = self.rgb_img(level, (randx, randy), size)
					file_name = self.img_path.split('/')
					file_name = file_name[len(file_name) - 1]
					file_name = file_name[:len(file_name) - 4]
					file_name = str(size[0]) + '_' + str(size[1]) + '_' + file_name
					file_name = file_name + '_' + str(current_normal) + '_0.png'
					normal_img.save(target_path + file_name)

					if current_normal % 1000 == 0:
						print "current: normal %r, pathological %r" % (current_normal, current_pathological)
					elif current_normal == normal_num:
						print "current: normal %r, pathological %r" % (current_normal, current_pathological)
				

			#crop pathological
			#sample coordinate
			region_xy = self.pathological_region[randr]

			min_x = region_xy[0]
			min_y = region_xy[1]
			max_x = region_xy[2]
			max_y = region_xy[3]

			# print region_xy

			randx = int(np.random.ranf(1) * (max_x - min_x) + min_x)
			randy = int(np.random.ranf(1) * (max_y - min_y) + min_y)
			# print randr
			# print min_x, min_y, max_x, max_y
			# print randx, randy

			if current_pathological < pathological_num:
				pathological_mask = self.rgb_mask(level, (randx, randy), size)
				#covert to array
				ary = np.asarray(pathological_mask)
				a = np.sum(ary, 2)
				#black (0, 0, 0, 255)
				#white (255, 255, 255, 255) 
				if np.min(a) == 1020:
					current_pathological = current_pathological + 1

					pathological_img = self.rgb_img(level, (randx, randy), size)

					file_name = self.img_path.split('/')
					file_name = file_name[len(file_name) - 1]
					file_name = file_name[:len(file_name) - 4]
					file_name = str(size[0]) + '_' + str(size[1]) + '_' + file_name
					file_name = file_name + '_' + str(current_pathological) + '_1.png'
					pathological_img.save(target_path + file_name)

					if current_pathological % 1000 == 0:
						print "current: normal %r, pathological %r" % (current_normal, current_pathological)
					elif current_pathological == pathological_num:
						print "current: normal %r, pathological %r" % (current_normal, current_pathological)
			if current_normal == normal_num and current_pathological == pathological_num:
				finish = True 
			else:
				finish = False
		
class ImageCropper(object):
	def __init__(self, img_path, mask_path, xml_path, target_path):
		self.img_path = img_path
		if img_path[-1] != '/':
			self.img_path = img_path + '/'
		self.mask_path = mask_path
		if mask_path[-1] != '/':
			self.mask_path = mask_path + '/'
		self.xml_path = xml_path
		if xml_path[-1] != '/':
			self.xml_path = xml_path + '/'
		self.target_path = target_path
		if target_path[-1] != '/':
			self.target_path = target_path + '/'

		self.mask_dic = {}
		for f in os.listdir(mask_path):
			num = f.split('_')[1]
			self.mask_dic[num] = f

		self.xml_dic = {}
		for f in os.listdir(xml_path):
			num = f.split('_')[1]
			num = num[:len(num) - 4]
			self.xml_dic[num] = f

		# print self.mask_dic
		# print self.xml_dic

	def run(self, size, level, normal_num, pathological_num):
		for f in os.listdir(self.img_path):
			#key
			key = f.split('_')[1]
			key = key[:len(key) - 4]

			img_name = self.img_path + f
			mask_name = self.mask_path + self.mask_dic[key]
			xml_name = self.xml_path + self.xml_dic[key]
			osr = OpenSlideReader(img_name, mask_name, xml_name)
			osr.crop_img(size, level, normal_num, pathological_num, self.target_path)

if __name__ == "__main__":
	ic = ImageCropper(argv[1], argv[2], argv[3], argv[4])
	ic.run((128, 128), 0, int(argv[5]), int(argv[6]))
# test = OpenSlideReader('./img/Tumor_001.tif', './mask/Tumor_001_Mask.tif', './xml/Tumor_001.xml')
# test.crop_img((128,128), 1, 15, 15, './')
#python slide.py /data/images/pathology/CAMELYON16/TrainingData/Train_Tumor /data/images/pathology/CAMELYON16/TrainingData/Ground_Truth/Mask /data/images/pathology/CAMELYON16/TrainingData/Ground_Truth/XML ./CropImgs 27000 4500