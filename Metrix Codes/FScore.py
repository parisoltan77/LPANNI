#Programmed by Soltanzadeh

import time
import json

def fetchResultOverlappingNodes(filePath):
    nodeSet = set({})
    with open(filePath, 'r') as fileHandle:
        line = fileHandle.readline()
        while line:
            nodeIdList = line.split()
            if len(nodeIdList) > 2:
                nodeSet.add(nodeIdList[0])
            line = fileHandle.readline()
    return nodeSet

def calculateFScore(communityFilePath, resultFilePath):
    groundTruth = fetchResultOverlappingNodes(communityFilePath)
    predictNodes = fetchResultOverlappingNodes(resultFilePath)
    correctNodes = predictNodes.intersection(groundTruth)
    precision = len(correctNodes)*1.0/len(predictNodes)
    recall = len(correctNodes)*1.0/len(groundTruth)
    if precision+recall > 0:
        fScore = 2*precision*recall/(precision+recall)
    else:
        fScore = 0
    #print('Precision: %.5f' % precision)
    #print('Recall: %.5f' % recall)
    #print('F1 Score: %.5f' % f1Score)
    return fScore, precision, recall


def main():
    startTime = time.time()
    with open('FScore.json', 'w') as fileHandle:
        jsonObj = {}
        for i in [1000, 5000, 10000]:
            jsonObj[str(i)] = {}
            for j in [1, 3]:
                jsonObj[str(i)]['0.%d' % j] = {'FScore': [], 'Precision': [], 'Recall': []}
                for k in [2, 3, 4, 5, 6, 7, 8]:
                    resultFilePath = "N-%d-mu0.%d-om%d-result.txt" % (i, j, k)
                    communityFilePath = "N-%d-mu0.%d-om%d-community.txt" % (i, j, k)
                    fScore, precision, recall = calculateFScore(communityFilePath, resultFilePath)
                   
                    jsonObj[str(i)]['0.%d' % j]['FScore'].append(fScore)
                    jsonObj[str(i)]['0.%d' % j]['Precision'].append(precision)
                    jsonObj[str(i)]['0.%d' % j]['Recall'].append(recall)
        fileHandle.write(json.dumps(jsonObj))
    completeTime = time.time()
    print('Result Written in FScore.json')
    print('Running Time: %.3fs' % (completeTime-startTime))

if __name__ == "__main__":
    main()