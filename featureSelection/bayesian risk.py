import numpy as np 
import pandas as pd 
from sklearn import metrics
# -Paso 1 seleccionar cada feature y discretizarla en n valores posibles (64 podría ser un buen número)
# -Paso 2 verificar la posibilidad de identificar inconsitency instances
# **Paso 3 crear función de particionado en el espacio por feature
# **Paso 4 crear los conjuntos de instancias inconsistentes maximales
# Paso 5 verificar propiedad de monoticidad experimental


# Data binning, Discrete binning, Bucketing
def featureBinning(X, nbins=32):
	newX = np.zeros(X.shape)
	for i in range(X.shape[1]):
		c = X[:, i]
		bins = np.linspace(np.min(c), np.max(c), nbins)
		bx = np.digitize(X[:, i], bins)
		newX[:, i] = bx
	return newX


# Transform a row to a string (USED: for the key in the dictionary of inconsistency set)
def rowToString(f):
	key = np.array2string(f)
	chars = "[]"
	for char in chars:
		key = key.replace(char, "")
	key = key.replace(". ", " ")
	while "  " in key:
		key = key.replace("  ", " ")
	return key


def inconInstances(X, y):
	inset = {}
	# Construct inconsistency set
	for i in range(X.shape[0]):
		row = X[i, :]
		key = rowToString(row)
		if key in inset:
			if y[i] in inset[key]:
				inset[key][y[i]] += 1
			else:
				inset[key][y[i]] = 1
		else:
			inset[key] = {y[i]: 1}
	# Sum total of inconsistencies
	insetCount = 0
	# Explore the inconsistent instances and count the instances with lower frequency
	for k in inset:
		if len(inset[k]) > 1:
			minValue = 9999999
			minKey = -1
			incCount = 0
			for ik in inset[k]:
				incCount += inset[k][ik]
				# print(incCount,inset[k][ik])
				if minValue > inset[k][ik]:
					minValue = inset[k][ik]
					minKey = ik
			insetCount += minValue
	return insetCount/X.shape[0]


def rankingRelevance(X, y):
	scores = []
	for i in range(X.shape[1]):
		f = X[:, i]
		score = metrics.mutual_info_score(f, y)
		scores.append(score)
	scores = np.array(scores)
	ranking = np.argsort(scores)
	return list(ranking)


def interact(X, y, t=0.0005):
	featuresRanking = rankingRelevance(X, y)
	# featuresRanking.reverse()
	features = featuresRanking.copy()
	# print(featuresRanking)
	inScore = inconInstances(X, y)
	print(inScore)
	for i in range(len(featuresRanking)):
		features.remove(featuresRanking[-i-1])
		newInScore = inconInstances(X[:, features], y)
		dScore = newInScore-inScore
		if dScore > t:
			features.append(featuresRanking[-i-1])
		else:
			inScore = newInScore
		print(features, inScore, newInScore, dScore)
		# print(features,dScore)


# Reading Data
df = pd.read_csv('sc1.csv', sep=',')
df = df.drop(df.index[0:1])
df = df.sample(frac=1).reset_index(drop=True)
data = df.values
data = data[:20000, :]

# Pre-Processing Data
X = data[:, 0:-1]
print(X.shape)
X = X.astype(float)
y = data[:, -1]
X = featureBinning(X, 16)
# print(X)
interact(X, y)
np.savetxt('foo.csv', X.astype('uint8'), fmt='%i', delimiter=",")