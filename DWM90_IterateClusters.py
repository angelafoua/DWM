#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import DWM10_Parms
import DWM95_CalculateEntropy
def iterateClusters(clusterList, refDict, linkIndex):
    logFile = DWM10_Parms.logFile
    print('\n>>Starting DWM90')
    print('\n>>Starting DWM90', file=logFile)
    epsilon = DWM10_Parms.epsilon
    iterationLinkIndex = linkIndex.copy()
    refCnt = 0
    clusterCnt = 0
    clusterCnt2 = 0
    goodClusterCnt = 0
    goodRefsCnt = 0
    # cluster is a list to store one cluster
    cluster = []
    clusterIndex = []
    caboose = ('---','---')
    # Add caboose to signal end of list
    clusterList.append(caboose)
    # Iterate through cluster pairs, but not caboose
    for j in range(0,len(clusterList)-1):
        currentPair = clusterList[j]
        clusterID = currentPair[0]
        refID = currentPair[1]
        tokenList = refDict[refID].copy()
        # Append token string to cluster
        cluster.append(tokenList)
        clusterIndex.append(refID)
        nextPair = clusterList[j+1]
        currentCID = currentPair[0]
        nextCID = nextPair[0]
        # Look ahead to see if at end of cluster, if yes, process cluster
        if currentCID != nextCID:
            clusterCnt +=1
            if len(cluster)>1:
                quality = DWM95_CalculateEntropy.calculateEntropy(cluster)
                clusterCnt2 +=1
            else:
                quality = 1.0
            # only write good clusters to LinkIndex
            if quality >= epsilon:
                goodClusterCnt +=1
                goodRefsCnt +=len(cluster)
                for k in range(0,len(clusterIndex)):
                    indexVal = clusterIndex[k]
                    linkIndex[indexVal] = currentCID
            # write all cluster good or bad to iteration LinkIndex
            for k in range(0,len(clusterIndex)):
                refCnt +=1
                indexVal = clusterIndex[k]
                iterationLinkIndex[indexVal] = currentCID
            cluster.clear()
            clusterIndex.clear()
    print('Total Clusters Processed =',clusterCnt)
    print('Total Clusters Processed =',clusterCnt, file=logFile)
    print('Total References in Clusters =', refCnt)
    print('Total References in Clusters =', refCnt, file=logFile)
    print('Total Clusters Size>1 Processed =',clusterCnt2)
    print('Total Clusters Size>1 Processed =',clusterCnt2, file=logFile)
    print('Total Good Clusters =',goodClusterCnt,' at epsilon =', epsilon)
    print('Total Good Clusters =',goodClusterCnt,' at epsilon =', epsilon, file=logFile)
    print('Total References in Good Cluster =', goodRefsCnt)
    print('Total References in Good Cluster =', goodRefsCnt, file=logFile)
    return iterationLinkIndex

