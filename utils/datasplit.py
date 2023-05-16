import os
import pandas as pd
import numpy as np
import argparse
import shutil
import string



# def create_output(split,output):
# 	if not os.path.exists(output):
# 		os.mkdir(output)
# 	for s in split:
# 		folder_name = s.split('.')[0]
# 		folder = os.path.join(output, folder_name)
# 		if not os.path.exists(folder):
# 			os.mkdir(folder)



def main(input, split, output):
	images = os.listdir(input)
	splits = os.listdir(split)

	# create_output(splits, output)

	folders = {}

	for s in splits:
		with open(os.path.join(split, s)) as file:

			files = file.readlines()

			files = list(map(lambda x: x.split('/')[-1].replace('\n', ''), files))
			folders[s] = files

	for key in folders.keys():
		ignore = list(set(images).difference(set(folders[key])))
		shutil.copytree(input, os.path.join(output, key.split('.')[0]), ignore = lambda x,y: ignore)


	



if __name__ == "__main__":
	PATH = os.path.dirname(os.path.realpath(__file__))


	parser = argparse.ArgumentParser(
						prog = 'Data splitter',
						description='Splitting the data with source path i by split path s')

	parser.add_argument('-p', '--path', type = str, default= PATH)
	parser.add_argument('-i','--input' ,type = str, default = 'images')
	parser.add_argument('-o', '--output', type=str, default= 'output')
	parser.add_argument('-s', '--split', type=str, default= 'original_splits')
	args = parser.parse_args()

	input = os.path.join(args.path, args.input)
	output = os.path.join(args.path, args.output)
	split = os.path.join(args.path, args.split)

	main(input, split, output)