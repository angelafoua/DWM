#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import DWM10_Parms
def calculateEntropy(cluster):
    # Calculate Normalization Base
    tokenCount = 0
    refCnt = len(cluster)
    for j in range(0, refCnt):
        #print(cluster[j])
        tokenCount = tokenCount + len(cluster[j])
    baseProb = 1/float(refCnt)
    base = -tokenCount*baseProb*math.log(baseProb,2)
    #print("Cluster size=", refCnt," Token Count=", tokenCount," Base Prob=", baseProb, " base=", base)
    epsilon = DWM10_Parms.epsilon
    entropy = 0.0
    clusterSize = len(cluster)
    #print('cluster size =', clusterSize)
    for j in range(0, len(cluster)-1):
        jList = cluster[j]
        #print('j=',j,'jList=', jList)
        for token in jList:
            cnt = 1
            #print('token=', token, 'cnt=', cnt)
            for k in range(j+1,len(cluster)):
                #print('k=',k)
                if token in cluster[k]:
                    cnt +=1
                    cluster[k].remove(token)
                    #print('token found in ',k, cluster[k])
            tokenProb = cnt/clusterSize
            term = -tokenProb*math.log(tokenProb,2)
            entropy +=term
            quality = 1.0 - entropy/base
            if quality < epsilon:
                #print('quit early top row, entropy=', entropy, ' quality=',1-entropy/base)
                return quality
            #print('**token=',token,'tokenProb=',tokenProb,' term=', term, 'entropy=', entropy)
            cnt = 0
    # Finish up for any tokens left in the last reference of the cluster
    for token in cluster[clusterSize-1]:
        tokenProb = 1.0/clusterSize
        term = -tokenProb*math.log(tokenProb,2)
        entropy +=term
        quality = 1.0 - entropy/base
        if quality < epsilon:
            #print('quit early last row, entropy=', entropy,' normalized=',1-entropy/base)
            return quality
        #print('last row token=', token, 'tokenProb=',tokenProb,' term=', term, 'entropy=', entropy)
    #print('entire cluster scanned, entropy=', entropy, ' normalized=',1-entropy/base)
    quality = 1.0 - entropy/base
    return quality

