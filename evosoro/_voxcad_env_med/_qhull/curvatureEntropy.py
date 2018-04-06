import sys
import numpy as np
import math
import myUtils as util


if(len(sys.argv) < 2):
	print("[curvatureEntropy.py] Python entropy calculation: input curvatures file not passed as argument! Quitting!")
	exit(1)

r  = util.readSingleLineFromFile(sys.argv[1]) # single line of space separated values
vs = r.split()
values = map(float, vs)

#print values

numBins = 100
minVal  = -2*math.pi
maxVal  = 2*math.pi
entropy = util.computeEntropy(values, numBins, minVal, maxVal)

# Override tmp file 
f = open(sys.argv[1], 'w')
f.write(str(entropy))
f.write("\n")

