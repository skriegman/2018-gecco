import numpy as np
#import os

###############################################################################################
# Author: Francesco Corucci - f.corucci@sssup.it
# Description: This file contains utility functions to compute robot's symmetry descriptors
#		from its shapeMatrix. See the commented main for usage
###############################################################################################

# ------------------------------------------------------------------------------------
# WARNING: these functions have been design with a different convention 
# 	   with respect to the one used to define shapeMatrix.
#	   Nomenclature may be ambiguous (x,y,z, front side, top).
#	   The correct correspondeces are embedded in computeRobotSymmetryIndices 
# ------------------------------------------------------------------------------------
 
#def main():
#	os.system('clear') 
#
#	mat = userInputSquareMatrix()
#	(globalSymmetryIdx, xSymmetryIdxAggregated, ySymmetryIdxAggregated, zSymmetryIdxAggregated, xyzSymmetryIndices) = computeRobotSymmetryIndices(mat)
#	print
#	print 'All six symmetry descriptors computed in x,y,z directions (rows: x,y,z, cols: listed below)'
#	print
#	print 'VertRefl | VerTran | HorRefl | HorTran | diagRefl | antiDiagRefl'
#	print(xyzSymmetryIndices)
#	print	
#	print 'Global symmetry = ',str(globalSymmetryIdx)
#	print 'xSymmetry = ', str(xSymmetryIdxAggregated)
#	print 'ySymmetry  =',str(ySymmetryIdxAggregated)
#	print'zSymmetry = ', str(zSymmetryIdxAggregated)	
#	print

def computeRobotSymmetryIndices(mat):
	#
	# top: 	   along z
	# front:   along x
	# lateral: along y
	#
	try:
		m = extractNotNullSubmatrix(mat)

		directionalIndices = []
		xyzSymmetryIndices = np.zeros((3,6))

		#print '--------------------------------------------------------------------'	
		#print ' Front scan descriptors'
		#print
		frontSymmetryIndicesSlices = scanMatrixForSymmetriesFront(m)
		frontSymmetryIndices = aggregateSlicesSymmetryIndices(frontSymmetryIndicesSlices)
		xyzSymmetryIndices[0,:] = frontSymmetryIndices	
		#printSymmetryIndices(frontSymmetryIndices)
		frontSymmetryIndex = frontSymmetryIndices.mean()
		directionalIndices.append(frontSymmetryIndex)
		#print 'Global frontal symmetry index: '+str(frontSymmetryIndex)

		#print '--------------------------------------------------------------------'
		#print ' Side scan descriptors'
		#print
		sideSymmetryIndicesSlices = scanMatrixForSymmetriesSide(m)
		sideSymmetryIndices = aggregateSlicesSymmetryIndices(sideSymmetryIndicesSlices)
		xyzSymmetryIndices[1,:] = sideSymmetryIndices
		#printSymmetryIndices(sideSymmetryIndices)
		sideSymmetryIndex = sideSymmetryIndices.mean()
		directionalIndices.append(sideSymmetryIndex)
		#print 'Global side symmetry index: '+str(sideSymmetryIndex)

		#print '--------------------------------------------------------------------'
		#print ' Top scan descriptors'
		#print
		topSymmetryIndicesSlices = scanMatrixForSymmetriesTop(m)
		topSymmetryIndices = aggregateSlicesSymmetryIndices(topSymmetryIndicesSlices)
		xyzSymmetryIndices[2,:] = topSymmetryIndices
		#printSymmetryIndices(sideSymmetryIndices)
		topSymmetryIndex = topSymmetryIndices.mean()
		directionalIndices.append(topSymmetryIndex)
		#print 'Global top symmetry index: '+str(topSymmetryIndex)
		#print '--------------------------------------------------------------------'
		
		overallSymmetryIndex = np.mean(directionalIndices)
		#print 'OVERALL SYMMETRY INDEX: '+str(overallSymmetryIndex)
		#print '--------------------------------------------------------------------'	

		# Fixing conventions	
		globalSymmetryIdx = overallSymmetryIndex
		xSymmetryIdx = frontSymmetryIndex
		ySymmetryIdx = sideSymmetryIndex
		zSymmetryIdx = topSymmetryIndex

		return (globalSymmetryIdx, xSymmetryIdx, ySymmetryIdx, zSymmetryIdx, xyzSymmetryIndices)
	
	except Exception, e:  
		#print("computeRobotSymmetryIndices: Couldn't compute symmetry indices (probably empty matrix). Will return all ones. Msg: %s"%e)
		return (1.0, 1.0, 1.0, 1.0, 1.0)


def scanMatrixForSymmetriesFront(m): # navigate 3D matrix from top to bottom
	# m[z,i,j]
	dim = np.shape(m)
	allSlicesIndices = np.zeros((dim[0], 6)) # 6 symmetry descriptors are currently defined
	#print 'Front view'
	for z in range(dim[0]):	
		thisSlice = m[z,:,:]
		allSlicesIndices[z, :] = computeAllSymmetryIndices(thisSlice)
		#print str(thisSlice)
	return allSlicesIndices

def scanMatrixForSymmetriesSide(m): # navigate 3D matrix from right to left
	# m[j,i,z]
	dim = np.shape(m)
	allSlicesIndices = np.zeros((dim[2], 6))
	count = 0
	#print 'Side view'
	for z in list(reversed(range(dim[2]))):
		thisSlice = np.transpose(m[:,:,z])
		allSlicesIndices[count, :] = computeAllSymmetryIndices(thisSlice)
		count = count + 1
		#print str(thisSlice)
	return allSlicesIndices

def scanMatrixForSymmetriesTop(m): # navigate 3D matrix from top to bottom
	# m[i,z,j]
	dim = np.shape(m)
	allSlicesIndices = np.zeros((dim[1], 6))
	#print 'Top view'
 	for z in range(dim[1]):
		thisSlice = np.flipud(m[:,z,:])
		allSlicesIndices[z, :] = computeAllSymmetryIndices(thisSlice)
		#print str(thisSlice)
	return allSlicesIndices

def indicesAreOk(i1, j1, i2, j2, dim1, dim2):
	return (i1 >= 0 and i2 >= 0 and j1 >= 0 and j2 >= 0) and (i1 < dim1 and i2 < dim1) and (j1 < dim2 and j2 < dim2);

def overlapIndex(overlap, nonOverlap):
	if (overlap == 0 and nonOverlap == 0): # handles special cases
		return 1

	return float(overlap)/(overlap+nonOverlap)

def printMat(m):
	print
	dim = np.shape(m)
	print 'Matrix dimension: '+str(dim)
	print m
	print	

def userInputSquareMatrix():
	
	dimRaw = raw_input('> Matrix dimensions (e.g. 3x3x3): ')
	dimRaw = dimRaw.split('x')
	dim = map(int, dimRaw)

	dimX = dim[0]
	dimY = dim[1]
	dimZ = dim[2]
	
	mat = np.zeros((dimZ, dimX, dimY)); #, dtype=np.int32)


	for i in range(dimZ): # iterate over slices of the 3D matrix
		print('Insert '+str(dimX)+'x'+str(dimY)+' matrix ('+str(i+1)+'/'+str(dimZ) +'):')
		for j in range(dimX):
			line = raw_input()
			line = line.split()
			values = map(float, line)

			if len(values) != dimY:
				print 'Wrong number of arguments provided. Expecting '+str(dimY)+', received '+str(len(values))
				print 'Quitting.'
				quit()

			mat[i,j,:] = values
		print
	return mat		

def extractNotNullSubmatrix(m):
	# Works with planar and cubic matrices
	nonZero = np.nonzero(m)
	L = np.shape(nonZero)[0]
	maxId = np.zeros(L)
	minId = np.zeros(L)
	
	for i in range(L):
		maxId[i] = np.max(nonZero[i])
		minId[i] = np.min(nonZero[i])	

	newMat = m[minId[0]:maxId[0]+1, minId[1]:maxId[1]+1, minId[2]:maxId[2]+1]	

	return newMat

def verticalReflectSymmetry(m):
	dim = np.shape(m)
	rows = dim[0]
	cols = dim[1]
	
	overlap = 0
	nonOverlap = 0

	for i in range(rows):
		for j in range(cols):
			j1 = j
			j2 = cols-j-1
			if (not indicesAreOk(i, j1, i, j2, rows, cols)) or (j1 >= j2):
				break
			if np.allclose(m[i,j1], m[i,j2]): # this way it'll work with floats as well. See additional arguments to set the absolute and tolerance
				overlap = overlap + 1
			else:
				nonOverlap = nonOverlap + 1
	
	if (overlap == 0 and nonOverlap == 0):
		return (rows*cols, 0)

	return (overlap, nonOverlap)

def verticalTranslationSymmetry(m):
	dim = np.shape(m)
	rows = dim[0]
	cols = dim[1]
	
	overlap = 0
	nonOverlap = 0

	for i in range(rows):
		for j in range(cols):
			j1 = j
			j2 = np.ceil(float(cols)/2)+j
	
			if (not indicesAreOk(i, j1, i, j2, rows, cols)):
				break
	
			if np.allclose(m[i,j1], m[i,j2]): # this way it'll work with floats as well. See additional arguments to set the absolute and tolerance
				overlap = overlap + 1
			else:
				nonOverlap = nonOverlap + 1
	if (overlap == 0 and nonOverlap == 0):
		if cols < 2:
			return (0, rows*cols)
		else:
			return (rows*cols, 0)

	return (overlap, nonOverlap)


def horizontalReflectSymmetry(m):
	dim = np.shape(m)
	rows = dim[0]
	cols = dim[1]	
	overlap = 0
	nonOverlap = 0

	for j in range(cols):
		for i in range(rows):
			i1 = i
			i2 = rows-i-1
			if (not indicesAreOk(i1, j, i2, j, rows, cols)) or (i1 >= i2):
				break

			if np.allclose(m[i1,j], m[i2,j]):
				overlap = overlap + 1
			else:
				nonOverlap = nonOverlap + 1
	if (overlap == 0 and nonOverlap == 0):
		return (rows*cols, 0)

	return (overlap, nonOverlap)

def horizontalTranslationSymmetry(m):
	dim = np.shape(m)
	rows = dim[0]
	cols = dim[1]
	
	overlap = 0
	nonOverlap = 0

	for j in range(cols):
		for i in range(rows):		
			i1 = i
			i2 = np.ceil(float(rows)/2)+i
	
			if (not indicesAreOk(i1, j, i2, j, rows, cols)):
				break
	
			if np.allclose(m[i1,j], m[i2,j]): # this way it'll work with floats as well. See additional arguments to set the absolute and tolerance
				overlap = overlap + 1
			else:
				nonOverlap = nonOverlap + 1
	if (overlap == 0 and nonOverlap == 0):
		if cols < 2:
			return (0, rows*cols)
		else:
			return (rows*cols, 0)

	return (overlap, nonOverlap)

def diagonalReflectSymmetry(m):
	dim = np.shape(m)
	rows = dim[0]
	cols = dim[1]
	
	overlap = 0
	nonOverlap = 0

	for i in range(rows):
		for j in range(cols):		
			i2 = j
			j2 = i
	
			if (not indicesAreOk(i, j, i2, j2, rows, cols)) or (i == j):
				break	

#			print '('+str(i)+','+str(j)+') - ('+str(i2)+','+str(j2)+')'	
			if np.allclose(m[i,j], m[i2,j2]): # this way it'll work with floats as well. See additional arguments to set the absolute and tolerance
				overlap = overlap + 1
			else:
				nonOverlap = nonOverlap + 1
	if (overlap == 0 and nonOverlap == 0):
		if (rows == cols):
			return (rows*cols, 0)
		else:
			return (0, rows*cols)

	return (overlap, nonOverlap)

def antiDiagonalReflectSymmetry(m):
	dim = np.shape(m)
	rows = dim[0]
	cols = dim[1]
	
	overlap = 0
	nonOverlap = 0

	for i in range(rows):
		for j in range(cols):		
			i2 = rows-j-1
			j2 = cols-i-1
	
			if (not indicesAreOk(i, j, i2, j2, rows, cols)) or (i == i2 or j == j2):
				break	

			if np.allclose(m[i,j], m[i2,j2]): # this way it'll work with floats as well. See additional arguments to set the absolute and tolerance
				overlap = overlap + 1
			else:
				nonOverlap = nonOverlap + 1
	if (overlap == 0 and nonOverlap == 0):
		if (rows == cols):
			return (rows*cols, 0)
		else:
			return (0, rows*cols)

	return (overlap, nonOverlap)

def aggregateSlicesSymmetryIndices(indices):
	# compute each single symmetry descriptor for the robot by taking the mean of
	# descriptor value of all matrix slices

	# descriptors: columns
	# observations: rows
	return indices.mean(0) # mean by columns

def sliceGlobalSymmetryIndex(symmetryIndices):	
	# compute a global symmetry index for a slice of the 3D matrix describing the robot
	# by just averaging all the symmetry descriptors
	return	np.mean(symmetryIndices)

def computeAllSymmetryIndices(m):
	symmetryIndices = []

	(overlap, nonOverlap) = verticalReflectSymmetry(m)
	symmetryIndices.append(overlapIndex(overlap, nonOverlap))

	(overlap, nonOverlap) = verticalTranslationSymmetry(m)
	symmetryIndices.append(overlapIndex(overlap, nonOverlap))

	(overlap, nonOverlap) = horizontalReflectSymmetry(m)
	symmetryIndices.append(overlapIndex(overlap, nonOverlap))

	(overlap, nonOverlap) = horizontalTranslationSymmetry(m)
	symmetryIndices.append(overlapIndex(overlap, nonOverlap))

	(overlap, nonOverlap) = diagonalReflectSymmetry(m)
	symmetryIndices.append(overlapIndex(overlap, nonOverlap))

	(overlap, nonOverlap) = antiDiagonalReflectSymmetry(m)
	symmetryIndices.append(overlapIndex(overlap, nonOverlap))

	return symmetryIndices

def printSymmetryIndices(l):
	print 'VertRefl | VerTran | HorRefl | HorTran | diagRefl | antiDiagRefl'
	print l 


#if __name__ == "__main__":
#	main()

