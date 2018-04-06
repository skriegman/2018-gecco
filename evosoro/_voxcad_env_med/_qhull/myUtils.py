import numpy as np
import math
import sys

def computeEntropy(data, numBins, minVal, maxVal): # Shannon entropy on input data. Expects a np array or a matrix (automatically flattened)
	hist = np.histogram(data,bins=numBins,range=(minVal, maxVal))[0]
	normed_hist = [x/float(len(data)) for x in hist]
	return -1.0 * sum([x*np.log2(x) for x in normed_hist if x>0])

def readSingleLineFromFile(filename):
	maxAttempts = 60
	attempt = 0
	result = -1
	
	while attempt < maxAttempts:
		attempt = attempt +1

		try:
			f = open(filename, "r")
			line = f.readline() # read a single line from file	
			result = line.strip()
			f.close()
		except IOError:
			continue
		else:
			break

	if attempt == maxAttempts:
		print '[ERROR myUtils.py readSingleLineFromFile] Cannot read file %s in %d attempts!'%(filename,maxAttempts)
		quit()

	return result # removes trailing \n

