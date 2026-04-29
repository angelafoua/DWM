#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import time
import datetime
from csv import reader
import DWM10_Parms
import DWM100_ReportData

def generateMetrics(linkIndex):
    logFile = DWM10_Parms.logFile
    print('\n>>Starting DWM99')
    print('>>Starting DWM99', file=logFile)
    truthFileName = DWM10_Parms.truthFileName
    print('Truth File Name=', truthFileName)
    print('Truth File Name=', truthFileName, file=logFile)    
    def countPairs(dict):
        totalPairs = 0
        for cnt in dict.values():
            pairs = cnt*(cnt-1)/2
            totalPairs +=pairs
        return totalPairs
    erDict = {}
    for refID in linkIndex:
        clusterID = linkIndex[refID]
        erDict[refID] = (clusterID,'x')
    truthFile = open(truthFileName,'r')
    line = (truthFile.readline()).strip()
    line = (truthFile.readline()).strip()
    while line != '':
        part = line.split(',')
        recID = part[0].strip()
        truthID = part[1].strip()
        if recID in erDict:
            oldPair = erDict[recID]
            clusterID = oldPair[0]
            newPair = (clusterID, truthID)
            erDict[recID] = newPair
        line = (truthFile.readline()).strip()
    linkedPairs = {}
    equivPairs = {}
    truePos = {}
    clusterIndex = []
    for pair in erDict.values():
        clusterID = pair[0]
        truthID = pair[1]
        if pair in truePos:
            cnt = truePos[pair]
            aPair = [pair[0],truthID]
            clusterIndex.append(aPair)                        
            cnt +=1
            truePos[pair] = cnt            
        else:
            truePos[pair] = 1
        if clusterID in linkedPairs:
            cnt = linkedPairs[clusterID]
            cnt +=1
            linkedPairs[clusterID] = cnt
        else:
            linkedPairs[clusterID] = 1
        if truthID in equivPairs:
            cnt = equivPairs[truthID]
            cnt +=1
            equivPairs[truthID] = cnt
        else:
            equivPairs[truthID] = 1   
    # End of counts
    L = countPairs(linkedPairs)
    E = countPairs(equivPairs)
    TP = countPairs(truePos)
    FP = float(L-TP)
    FN = float(E-TP)
    if L > 0:
        precision = round(TP/float(L),4)
    else:
        precision = 1.00
    if E > 0:
        recall = round(TP/float(E),4)
    else:
        recall = 1.00
  
    fmeas = round((2*precision*recall)/(precision+recall),4)
      
    # for report process
    DWM10_Parms.precision = precision
    DWM10_Parms.recall = recall
    DWM10_Parms.fmeasure = fmeas
    DWM10_Parms.truePairs = TP
    DWM10_Parms.expectedPairs = E
    DWM10_Parms.linkedPairs = L
    
    print('True Pairs =',TP)
    print('True Pairs =',TP, file=logFile)
    print('Expected Pairs =',E)
    print('Expected Pairs =',E, file=logFile)
    print('Linked Pairs =',L)
    print('Linked Pairs =',L, file=logFile)
    print('Precision=',precision)
    print('Precision=',precision, file=logFile)
    print('Recall=', recall)
    print('Recall=', recall, file=logFile)
    print('F-measure=', fmeas)
    print('F-measure=', fmeas, file=logFile)
    
    return

