#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import os
import csv
import sys
import time
from datetime import datetime
import DWM10_Parms
from textdistance import Levenshtein
from textdistance import DamerauLevenshtein
changeCount = 0


# In[2]:


def normalLED(token1,token2):
    Class = DamerauLevenshtein()
    length = max(len(token1), len(token2))
    # normalize by length, high score wins
    dDist = Class.distance(token1,token2)
    fDist = float(length - dDist)/ float(length);
    return fDist, dDist


# In[3]:


def incTokenFreq(token,freqDict):
    if token in freqDict:
        freqDict[token] += 1
    else:
        freqDict[token] = 1


# In[ ]:


def decTokenFreq(token,freqDict):
    if token in freqDict:
        if freqDict[token] == 1:
            del freqDict[token]
        else:
            freqDict[token] =- 1


# In[4]:


def isAlias(token1,token2,aliasDict):
    if token1 in aliasDict:
        if aliasDict[token1].lower() == token2.lower():
            return True
        else:
            return False
    else:
        return False


# In[5]:


def tokenLogicNew(rowjID, rowkID, index, logFile, aliasDict, refDict):
    changeDict={}
    rowjTokens = refDict[rowjID]
    rowkTokens = refDict[rowkID]
    kRefID= rowkID
    kGroupID = "merged"
    jRefID= rowjID
    jGroupID = "merged"
    if rowjTokens != rowkTokens:
        sizej =len(rowjTokens)
        sizek =len(rowkTokens)
        rowJ= " ".join(rowjTokens)
        rowK= " ".join(rowkTokens)
        global changeCount
        indexJinc = 0
        mothod=1
        oldJ = None
        oldJ1 = None
        oldJ2 = None
        for indexJ, j in enumerate(rowjTokens):
            oldK = None
            oldK1 = None
            oldK2 = None
            for indexK, k in enumerate(rowkTokens):
                if j==k:
                    if indexJ+2 < sizej:
                        tokenJ= j
                        tokenJ1= rowjTokens[indexJ+1]
                        tokenJ2= rowjTokens[indexJ+2]
                        lenTokenJ = len(tokenJ)
                        lenTokenJ1 = len(tokenJ1)
                    else:
                        continue
                    if indexK+2 < sizek:
                        tokenK= k
                        tokenK1= rowkTokens[indexK+1]
                        tokenK2= rowkTokens[indexK+2]
                        lenTokenK = len(tokenK)
                        lenTokenK1 = len(tokenK1)                            
                    else:
                        continue
                    if tokenJ == tokenJ1:
                        continue
                    if tokenK == tokenK1:
                        continue
                    if oldJ == tokenJ and oldJ1 == tokenJ1 and oldJ2 ==tokenJ2:
                        continue
                    else:
                        oldJ = tokenJ
                        oldJ1 = tokenJ1 
                        oldJ2 =tokenJ2
                    if oldK == tokenK and oldK1 == tokenK1 and oldK2 ==tokenK2:
                        continue
                    else:
                        oldK = tokenK
                        oldK1 = tokenK1 
                        oldK2 =tokenK2
                    if tokenJ2 == tokenK2:
                        if tokenJ1 != tokenK1 and lenTokenK1 > 2 and lenTokenJ1 >2:
                            dist, dDist = normalLED(tokenJ1,tokenK1)
                            if dDist == 1:
                                freqjToken = index[tokenJ1]
                                freqkToken = index[tokenK1]
                                if freqjToken < freqkToken:
                                    rowjTokens[indexJ+1] = tokenK1
                                    changeCount+=1
                                    changeDict[str(tokenJ1)+","+str(tokenK1)]=changeDict.get(str(tokenJ1)+","+str(tokenK1),0)+1
                                    incTokenFreq(tokenK1,index)
                                    tokenJ1 = tokenK1
                                elif freqjToken > freqkToken:
                                    rowkTokens[indexK+1] = tokenJ1
                                    changeCount+=1
                                    changeDict[str(tokenK1)+","+str(tokenJ1)]=changeDict.get(str(tokenK1)+","+str(tokenJ1),0)+1
                                    incTokenFreq(tokenJ1,index)
                        elif tokenJ2 == tokenK2 and isAlias(tokenJ1,tokenK1,aliasDict) == True:
                            rowjTokens[indexJ+1] = tokenK1
                            changeCount+=1
                            changeDict[str(tokenJ1)+","+str(tokenK1)]=changeDict.get(str(tokenJ1)+","+str(tokenK1),0)+1
                            incTokenFreq(tokenK1,index)
                        elif tokenJ2 == tokenK2 and isAlias(tokenK1,tokenJ1,aliasDict) == True:
                            rowkTokens[indexK+1] = tokenJ1
                            changeCount+=1
                            changeDict[str(tokenK1)+","+str(tokenJ1)]=changeDict.get(str(tokenK1)+","+str(tokenJ1),0)+1
                            incTokenFreq(tokenJ1,index)
                    elif tokenJ == tokenK and tokenJ1 == tokenK2 and tokenJ2 != tokenK1: 
                        lenTokenJ2 = len(tokenJ2)
                        if lenTokenJ2 > 2 and lenTokenK1 >2:
                            rowjTokens.insert(indexJ+1, tokenK1)
                            changeCount+=1
                            sizej=len(rowjTokens)
                            changeDict[","+str(tokenK1)]=changeDict.get(","+str(tokenK1),0)+1
                            incTokenFreq(tokenK1,index)
                    elif tokenJ == tokenK and tokenJ2 == tokenK1 and tokenJ1 != tokenK2: 
                        lenTokenK2 = len(tokenK2)
                        if lenTokenJ1 > 2 and lenTokenK2 >2:
                            rowkTokens.insert(indexK+1, tokenJ1)
                            changeCount+=1
                            sizek=len(rowkTokens)
                            changeDict[","+str(tokenJ1)]=changeDict.get(","+str(tokenJ1),0)+1
                            incTokenFreq(tokenJ1,index)
                    else:
                        inc = 1
                        tokenK1index=[]
                        tokenK1List=[]
                        tokenK1List.append(tokenK1)
                        tokenK1index.append(int(indexK+1))
                        for indexK2, k2 in enumerate(rowkTokens):
                            if indexK2 < indexK+2:
                                continue
                            if indexK2+inc+2 < sizek:
                                tokenK= tokenK
                                tokenK1= tokenK1+tokenK2
                                tokenK1List.append(tokenK2)
                                tokenK1index.append(int(indexK2))
                                tokenK2= rowkTokens[indexK2+inc]
                                if tokenJ2 == tokenK2:
                                    if tokenJ1 == tokenK1:
                                        if tokenK1.isdigit():
                                            rowkTokens[tokenK1index[0]] = tokenJ1
                                            for t in tokenK1index[1:]:
                                                del rowkTokens[t]
                                            changeCount+=1
                                            sizek=len(rowkTokens)
                                            changeDict[str(tokenK1)+","+str(tokenJ1)]=changeDict.get(str(tokenK1)+","+str(tokenJ1),0)+1
                                            incTokenFreq(tokenJ1,index)
                                            break
                                        else:
                                            rowjTokens[indexJ+1]=tokenK1List[0]
                                            tokenK1List.pop(0)
                                            newInc = 2
                                            for t in tokenK1List:
                                                rowjTokens.insert(indexJ+newInc, t)
                                                newInc += 1
                                            changeCount+=1
                                            sizej=len(rowjTokens)
                                            changeDict[str(tokenK1)+","+str(tokenJ1)]=changeDict.get(str(tokenK1)+","+str(tokenJ1),0)+1
                                            incTokenFreq(tokenJ1,index)
                                            break
                                else:
                                    continue        
                else:
                    if j.isnumeric() and k.isnumeric():
                        continue
                    if indexJ+2 < sizej and indexK+2 < sizek:
                        if len(j) > 2 and len(k) > 2:
                            if rowjTokens[indexJ+1] == rowkTokens[indexK+1]:
                                if rowjTokens[indexJ+2] == rowkTokens[indexK+2]: 
                                    dist, dDist = normalLED(j,k)
                                    if dDist == 1:
                                        freqjToken = index[j]
                                        freqkToken = index[k]
                                        if freqjToken < freqkToken:
                                            rowjTokens[indexJ]=k
                                            tokenJ = k
                                            changeDict[str(j)+","+str(k)]=changeDict.get(str(j)+","+str(k),0)+1
                                            incTokenFreq(str(k),index)
                                        else:
                                            rowkTokens[indexK]=j
                                            tokenK = j
                                            changeDict[str(k)+","+str(j)]=changeDict.get(str(k)+","+str(j),0)+1
                                            incTokenFreq(str(j),index)
                                        changeCount+=1
                                    else:
                                        continue
                                else:
                                    continue
                            else:
                                continue
                        else:
                            continue
                    else:
                        continue
    #This updates the refDict with updated tokens from RowJ and rowK
    refDict[jRefID]=rowjTokens
    refDict[kRefID]=rowkTokens
    return changeDict
    


# In[2]:


def RunBlockCorrections(blockPairList, blockFreqDict, refDict): #bl = blockMap
    logFile = DWM10_Parms.logFile
    print('\n>>Starting DWM45 - blockCorrection is set to True')
    print('\n>>Starting DWM45 - blockCorrection is set to True', file=logFile)
    global changeCount
    changeCount = 0
    logFile = DWM10_Parms.logFile
    changeDict={}
    totalChangeDict={}
    
    #Start Loop: load aliasDict from alias file
    aliasDict = {}
    with open('alias.dat', 'r') as aliasFile:
        reader = csv.DictReader(aliasFile, delimiter='\t', fieldnames=['value', 'alias'])
        for line in reader:
            aliasDict[line['alias']]=line['value']
    aliasFile.close()
    #End Loop: Load aliasDict
    
    #itterate over blockPairList and cleanse each ref pair in list
    for line in blockPairList:
        line = line.split('|')
        #if the reference IDs are the same, then do not apply corrections, move to next pair
        if line[0].strip() == line[1].strip():
            continue
        refJID=line[0]
        refKID=line[1]
        changeDict=tokenLogicNew(refJID, refKID, blockFreqDict, logFile, aliasDict, refDict)
        #append changes to total dict and sum change count
        totalChangeDict= {k: totalChangeDict.get(k, 0) + changeDict.get(k, 0) for k in set(totalChangeDict) | set(changeDict)}
    if DWM10_Parms.blockCorrectionDetail:
        print('>>List of Block Corrections sent to logFile')
        print('>>List of Block Corrections - blockCorrectionDetail = True', file=logFile)
        for key in totalChangeDict:
            print(str(key),file=logFile)
    print("Block Token Corrections="+str(changeCount))
    print("Block Token Corrections="+str(changeCount), file=logFile)
    return changeCount


# In[ ]:




