import numpy as np
import random
import os

from . import Helper

def create_masks(saliency_methods, saliency_dir, mask_dir, model_type, model_name):
	percentages = [i for i in range(10, 91, 10)]

	for saliency in saliency_methods:
		if (saliency != "Random"):
			saliency_ = np.load(saliency_dir + model_name + "_" + model_type + "_" + saliency + "_rescaled.npy")

		else:
			randomSaliencyIndex = random.randint(0, len(saliency_methods) - 2)
			saliency_ = np.load(saliency_dir + model_name + "_" + model_type + "_" + saliency_methods[
				randomSaliencyIndex] + "_rescaled.npy")
			np.random.shuffle(np.transpose(saliency_))
			np.save(saliency_dir + model_name + "_" + model_type + "_" + saliency + "_rescaled", saliency_)

		saliency_ = saliency_.reshape(saliency_.shape[0], -1)
		indexGrid = np.zeros((saliency_.shape[0], saliency_.shape[1], len(percentages)), dtype='object')
		indexGrid[:, :, :] = np.nan
		for i in range(saliency_.shape[0]):
			indexes = Helper.getIndexOfAllhighestSalientValues(saliency_[i, :], percentages)
			for l in range(len(indexes)):
				indexGrid[i, :len(indexes[l]), l] = indexes[l]
		for p, percentage in enumerate(percentages):
			np.save(mask_dir + model_name + "_" + model_type + "_" + saliency + "_" + str(
				percentage) + "_percentSal_rescaled", indexGrid[:, :, p])

	print("Creating Masks for ", model_name + "_" + model_type)


def main(args,DatasetsTypes,DataGenerationTypes,models):
	if  os.path.exists(args.ignore_list):
		f = open(args.ignore_list, 'r+')
		ignore_list = [line for line in f.readlines()]
		f.close()
		for i in range(len(ignore_list)):
			if('\n' in ignore_list[i]):
				ignore_list[i]=ignore_list[i][:-1]
	else:
		ignore_list=[]

	saliency_methods = Helper.getSaliencyMethodsFromArgs(args)
	saliency_methods.append("Random")

	for x in range(len(DatasetsTypes)):
		for y in range(len(DataGenerationTypes)):

			if(DataGenerationTypes[y]==None):
				args.DataName=DatasetsTypes[x]+"_Box"
			else:
				args.DataName=DatasetsTypes[x]+"_"+DataGenerationTypes[y]

			modelName="Simulated"
			modelName+=args.DataName

			for m in range(len(models)):
				if(args.DataName+"_"+models[m] in ignore_list):
					print("ignoring",args.DataName+"_"+models[m]  )
					continue
				else:
					create_masks(saliency_methods, args.Saliency_dir, args.Mask_dir, models[m], modelName)
