#!/usr/bin/env python
# coding: utf-8

# In[1]:


import DWM10_Parms
from collections import OrderedDict
from textdistance import DamerauLevenshtein
#from textdistance import Levenshtein
import Levenshtein as lev
import operator


# In[2]:


def globalReplace(refDict, tokenFreqDict):
    logFile = DWM10_Parms.logFile
    print ("\n>>Starting DWM25 --- runGlobalCorrection is set to True")
    print("\n>>Starting DWM25 --- runGlobalCorrection is set to True", file=logFile)
    Class = DamerauLevenshtein()
#Phase 1 Create Dictionary from DWM_WordList
    wordListDict = {}
    wordListFile = open('DWM_WordList.txt','r')
    word = wordListFile.readline().strip()
    itemsRead = 1
    while word != '':
        wordListDict.update({word:''})
        word = wordListFile.readline().strip()
        itemsRead +=1
    wordListFile.close()
    print('Items read =', itemsRead)
    print('wordListDict size =', len(wordListDict))
    minFreqStdToken = DWM10_Parms.minFreqStdToken
    minLenStdToken = DWM10_Parms.minLenStdToken
    maxFreqErrToken = DWM10_Parms.maxFreqErrToken
    print("DWM_WordList loaded, word count = ", len(wordListDict))
    print("DWM_WordList loaded, word count = ", len(wordListDict), file=logFile)
    print("Minimum Frequency of Standard Token = ", minFreqStdToken)
    print("Minimum Frequency of Standard Token = ", minFreqStdToken, file=logFile)   
    print("Minimum Length of Standard Token = ", minLenStdToken)
    print("Minimum Length of Standard Token = ", minLenStdToken, file=logFile)   
    print("Maximum Frequency of Error Token = ", maxFreqErrToken)
    print("Maximum Frequency of Error Token = ", maxFreqErrToken, file=logFile)
#Phase 2, build list of token-frequency pairs and sort descending by token frequency
    print('tokenFreqDict size = ', len(tokenFreqDict))
    print('tokenFreqDict size = ', len(tokenFreqDict), file=logFile)
    sortedIndex = sorted(tokenFreqDict.items(),reverse=True, key=operator.itemgetter(1))
    tokenCnt = len(sortedIndex)
    print("Sorted Token Size =", tokenCnt)
    print("Sorted Token Size =", tokenCnt, file=logFile)
    cleanIndex = []
    for j in range(0,tokenCnt):
        pairJ = sortedIndex[j]
        wordJ = pairJ[0]
        lenJ = len(wordJ)
        freqJ = pairJ[1]
        if lenJ<minLenStdToken:
            continue
        if not wordJ.isalpha():
            continue
        if (freqJ<=maxFreqErrToken) and (wordJ in wordListDict):
            continue        
        cleanIndex.append(pairJ)
    cleanCnt = len(cleanIndex)
    print("Clean Token Size =", cleanCnt)
    print("Clean Token Size =", cleanCnt, file=logFile)
#Phase 3 Populate Dictionary (stdTokenDict) of token corrections
    stdTokenDict = {}
    checkCnt = 0
    for j in range(0,cleanCnt-1):
        pairJ = cleanIndex[j]
        wordJ = pairJ[0]
        lenJ = len(wordJ)
        freqJ = pairJ[1]
        if freqJ < minFreqStdToken:
            break
        for k in range(cleanCnt-1, 1, -1):
            pairK = cleanIndex[k]
            wordK = pairK[0]
            lenK = len(wordK)
            freqK = pairK[1]
            if freqK > maxFreqErrToken:
                break
            dis = lev.distance(wordJ.lower(),wordK.lower())     
            if dis == 1:
                stdTokenDict[wordK] = wordJ
                cleanIndex[k] = ('',freqK)                   
            elif dis == 2:
                if Class.distance(wordJ,wordK)==1:
                    stdTokenDict[wordK] = wordJ
                    cleanIndex[k] = ('',freqK)
    print('\nTotal correction pairs = ', len(stdTokenDict)) 
    print('\nTotal correction pairs = ', len(stdTokenDict), file=logFile) 
    # If detail requested, write changes to run log
    if DWM10_Parms.globalCorrectionDetail:
        print('Details of correction sent to logFile')
        print('Error Token, Correction Token', file=logFile)
        for token in stdTokenDict:
            print(token+','+stdTokenDict[token], file=logFile)
    # Apply corrections to all references
    newList = []
    newDict = {}
    tokenChangeCnt = 0
    refChangeCnt = 0
    for refID in refDict:
        tokenList = refDict[refID]
        #print('**ref=',refID, 'before', tokenList)
        changeMade = False
        newList = []
        change = False
        for token in tokenList:
            if token in stdTokenDict:
                newList.append(stdTokenDict[token])
                tokenChangeCnt +=1
                change = True
            else:
                newList.append(token)
        newDict[refID] = newList
        if change:
            refChangeCnt +=1
    print('Total tokens corrected = ', tokenChangeCnt) 
    print('Total tokens corrected = ', tokenChangeCnt, file=logFile)
    print('Total references corrected = ', refChangeCnt) 
    print('Total references corrected = ', refChangeCnt, file=logFile)
    return newDict

