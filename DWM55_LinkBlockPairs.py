#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
from textdistance import Cosine
from textdistance import MongeElkan
import DWM10_Parms
import DWM65_ScoringMatrixStd
import DWM66_ScoringMatrixKris
def linkBlockPairs(blockPairList, refDict, tokenFreqDict): 
    logFile = DWM10_Parms.logFile
    sigma = DWM10_Parms.sigma
    removeDuplicateTokens = DWM10_Parms.removeDuplicateTokens
    removeExcludedBlkTokens = DWM10_Parms.removeExcludedBlkTokens
    minBlkTokenLen = DWM10_Parms.minBlkTokenLen
    excludeNumericBlocks = DWM10_Parms.excludeNumericBlocks
    print('\n>>Starting DWM55')
    print('\n>>Starting DWM55', file=logFile)
    print('Sigma =', sigma)
    print('Sigma =', sigma, file=logFile)    
    print('Remove Duplicate Tokens =', removeDuplicateTokens)
    print('Remove Duplicate Tokens =', removeDuplicateTokens, file=logFile)
    print('Remove Excluded Block Tokens =', removeExcludedBlkTokens)
    print('Remove Excluded Block Tokens =', removeExcludedBlkTokens, file=logFile)
    # Define nested function for removing stop words
    def removeStopWords(tokenList):
        newList = []
        #print('-- tokenList', tokenList)
        for token in tokenList:
            tokenLen = len(token)
            includeToken = True
            freq = tokenFreqDict[token]
            if freq>=sigma:
                includeToken = False
                #print('-- sigma rule', token, freq)
            if removeExcludedBlkTokens:
                if tokenLen < minBlkTokenLen:
                    includeToken = False
                    #print('-- min len rule', token, tokenLen)
                if token.isdigit() and excludeNumericBlocks:
                    includeToken = False
                    #print('-- number rule', token)
            if removeDuplicateTokens and (token in newList):
                includeToken = False
                #print('-- duplicate token rule', token)
            if includeToken:
                newList.append(token)
       # print('-- newList', newList)
        return newList
    # end of nexted fucntion
    # Check for valid comparator
    validComparator = False
    comparator = DWM10_Parms.comparator
    if comparator == 'MongeElkan':
        Class = MongeElkan()
        validComparator = True
    if comparator == 'Cosine':
        Class = Cosine()
        validComparator = True
    if comparator == 'ScoringMatrixStd':
        Class = DWM65_ScoringMatrixStd
        validComparator = True
    if comparator == 'ScoringMatrixKris':
        Class = DWM66_ScoringMatrixKris
        validComparator = True        
    if not validComparator:
        print('**Error: Invalid Comparator Value in Parms File', comparator)
        sys.exit()
    mu = DWM10_Parms.mu
    linkedPairList = []
    blockPairListLen = len(blockPairList)
    for j in range(0, blockPairListLen):
        pair = blockPairList[j]
        refIDs = pair.split('|')
        refID1 = refIDs[0]
        refID2 = refIDs[1]
        tokenList1 = removeStopWords(refDict[refID1])
        tokenList2 = removeStopWords(refDict[refID2])
        result = Class.normalized_similarity(tokenList1[:],tokenList2[:])
        if result >= mu:
            linkedPairList.append((refID1,refID2))
    print('Number of Pairs Linked =', len(linkedPairList), 'at mu=', mu)
    print('Number of Pairs Linked =', len(linkedPairList), 'at mu=', mu, file=logFile)
    return linkedPairList

