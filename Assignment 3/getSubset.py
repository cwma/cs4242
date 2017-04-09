import csv
import json
from collections import Counter

def getLabels():
    labels = {}
    with open("botlabels.csv", "r") as labelfile:
        labelcsv = csv.reader(labelfile)
        for num, line in enumerate(labelcsv):
            if num > 0:
                labels[line[0]] = line[1]
        return labels

def getSubset(labels):
	subset = {}
	count = Counter()
	for tweetid, label in labels.items():
		if count[label] < 100 and label != "spam-trick":
			count[label] += 1
			subset[tweetid] = label
	with open("subsetlabels.csv", "w", newline="\n") as newfile:
		newcsv = csv.writer(newfile)
		for tweetid, label in subset.items():
			label = label.rstrip("\n")
			newcsv.writerow([tweetid, label])

if __name__ == "__main__":
	getSubset(getLabels())