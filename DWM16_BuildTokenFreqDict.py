#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import sys
import numpy as np
import operator
import math
import DWM10_Parms

def buildTokenFreqDict(refDict):
    logFile = DWM10_Parms.logFile
    print('\n>> Starting DWM16')
    print('\n>> Starting DWM16', file=logFile)    
    tokenCnt = 0
    refCnt = 0
    numTokenCnt = 0
    tokenFreqDict = {}
    tokenLenDict = {}
    for key in refDict:
        refCnt +=1
        tokenList = refDict[key]
        tokenCnt = tokenCnt + len(tokenList)
        for j in range(0, len(tokenList)):
            token = tokenList[j]
            if token.isdigit():
                numTokenCnt += 1
            if token in tokenFreqDict:
                tokenFreqDict[token] += 1
            else:
                tokenFreqDict[token] = 1
            tokenLen = len(token)
            if tokenLen in tokenLenDict:
                tokenLenDict[tokenLen] += 1
            else:
                tokenLenDict[tokenLen] = 1
    
    print('Total References Read=',refCnt)
    print('Total References Read=',refCnt, file=logFile)    
    DWM10_Parms.refCnt=refCnt
    print('Total Tokens Found =',tokenCnt)
    print('Total Tokens Found =',tokenCnt, file=logFile)  
    DWM10_Parms.tokenCnt=tokenCnt
    uniqueTokenCnt = len(tokenFreqDict)
    print('Total Unique Tokens =', uniqueTokenCnt)
    print('Total Unique Tokens =', uniqueTokenCnt, file=logFile)     
    DWM10_Parms.uniqueTokenCnt=uniqueTokenCnt
    uniqueTokenRatio = uniqueTokenCnt/tokenCnt
    uniqueTokenRatio = round(uniqueTokenRatio, 4)
    print('Unique Token Ratio =',uniqueTokenRatio)
    print('Unique Token Ratio =',uniqueTokenRatio, file=logFile)    
    DWM10_Parms.uniqueTokenRatio=uniqueTokenRatio   
    print('Total Numeric Tokens Found =',numTokenCnt)
    print('Total Numeric Tokens Found =',numTokenCnt, file=logFile)    
    DWM10_Parms.numTokenCnt=numTokenCnt
    numTokenRatio = numTokenCnt/tokenCnt
    numTokenRatio = round(numTokenRatio, 4)
    print('Numeric Token Ratio =',numTokenRatio)
    print('Numeric Token Ratio =',numTokenRatio, file=logFile)    
    DWM10_Parms.numTokenRatio=numTokenRatio
    # Build a list of token frequencies from dictionary
    tokenFreqCount = list(tokenFreqDict.values())
    # numpy functions that directly calculates minimum, maximum, average and standard deviation
    minFreq = min(tokenFreqCount)
    print('Minimum Token Frequency =', str(minFreq))
    print('Minimum Token Frequency =', str(minFreq), file=logFile)    
    DWM10_Parms.minFreq=minFreq
    maxFreq = max(tokenFreqCount)
    print('Maximum Token Frequency =', str(maxFreq))
    print('Maximum Token Frequency =', str(maxFreq), file=logFile)    
    DWM10_Parms.maxFreq=maxFreq
    sortedIndex = sorted(tokenFreqDict.items(),reverse=True, key=operator.itemgetter(1))
    sortLen = len(sortedIndex)
    if sortLen<10:
        top = sortLen
    else:
        top = 10
    print('Top Ten Tokens by Freqency')
    for j in range(0,top):
        pairJ = sortedIndex[j]
        wordJ = pairJ[0]
        lenJ = len(wordJ)
        freqJ = pairJ[1]
        print('  Token=', wordJ, 'Frequency=', freqJ)
        print('  Token=', wordJ, 'Frequency=', freqJ, file=logFile)
    avgFreq = np.mean(np.array(tokenFreqCount))
    avgFreq = round(avgFreq, 4)
    print('Average Token Frequency =', str(avgFreq))
    print('Average Token Frequency =', str(avgFreq), file=logFile)    
    DWM10_Parms.avgFreq=avgFreq
    stdFreq = np.std(np.array(tokenFreqCount))
    stdFreq = round(stdFreq, 4)
    print('Standard Deviation of Token Frequency =', str(stdFreq))
    print('Standard Deviation of Token Frequency =', str(stdFreq), file=logFile)    
    DWM10_Parms.stdFreq=stdFreq
    # Calculations for token lengths from dictionary
    totalF = 0
    totalFxL = 0
    totalFxL2 = 0
    maxLen = 0
    minLen = 999
    for key in tokenLenDict:
        if key>maxLen:
            maxLen = key
        if key<minLen:
            minLen = key
        freq = tokenLenDict[key]
        totalF += freq
        totalFxL += freq*key
        totalFxL2 += freq*key*key
    avgLen = totalFxL/totalF
    avgLen = round(avgLen, 4)
    stdDevLen = (totalF*totalFxL2 - totalFxL*totalFxL)/(totalF*(totalF-1))
    stdDevLen = math.sqrt(stdDevLen)
    stdDevLen = round(stdDevLen, 4)
    print('Minimum Token Length =', str(minLen))
    print('Minimum Token Length =', str(minLen), file=logFile)    
    DWM10_Parms.minLen=minLen
    print('Maximum Token Length =', str(maxLen))
    print('Maximum Token Length =', str(maxLen), file=logFile)    
    DWM10_Parms.maxLen=maxLen    
    print('Average Token Length =', str(avgLen))
    print('Average Token Length =', str(avgLen), file=logFile)    
    DWM10_Parms.avgLen=avgLen
    print('Stardard Devation of Token Length =', str(stdDevLen))
    print('Stardard Devation of Token Length =', str(stdDevLen), file=logFile)    
    DWM10_Parms.stdDevLen=stdDevLen
    
    return tokenFreqDict


