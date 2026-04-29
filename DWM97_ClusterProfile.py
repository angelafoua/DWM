#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import DWM10_Parms
def generateProfile(linkIndex):
    logFile = DWM10_Parms.logFile
    print('\n>>Starting DWM97')
    print('\n>>Starting DWM97', file=logFile)
    profileDict = {}
    clusterSizeDict = {}
    for key in linkIndex:
        clusterKey = linkIndex[key]
        if clusterKey not in clusterSizeDict:
            clusterSizeDict[clusterKey] = 1
        else:
            cnt = clusterSizeDict[clusterKey]
            cnt +=1
            clusterSizeDict[clusterKey] = cnt
    for key in clusterSizeDict:
        clusterSize = clusterSizeDict[key]
        if clusterSize not in profileDict:
            profileDict[clusterSize] = 1
        else:
            cnt = profileDict[clusterSize]
            cnt +=1
            profileDict[clusterSize] = cnt
    print('\nCluster Profile')
    print('\nCluster Profile', file=logFile)
    print('Size\tCount')
    print('Size\tCount', file=logFile)
    total = 0
    for key in sorted(profileDict.keys()) :
        clusterTotal = key*profileDict[key]
        total +=clusterTotal
        print(key, '\t', profileDict[key], '\t', clusterTotal)
        print(key, '\t', profileDict[key], '\t', clusterTotal, file=logFile)
    print('\tTotal\t', total)
    print('\tTotal\t', total, file=logFile)     
    return

