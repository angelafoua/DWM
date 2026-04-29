#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
import sys
import numpy as np
import operator
import DWM10_Parms
def tokenizeInput():
    #***********Inner Function*******************************
    #Replace delimiter with blanks, then compress token by replacing non-word characters with null
    def tokenizerCompress(string):
        string = string.upper()
        string = string.replace(delimiter,' ')
        tokenList = re.split('[\s]+',string)
        newList = []
        for token in tokenList:
            newToken = re.sub('[\W]+','',token)
            if len(newToken)>0:
                newList.append(newToken)
        return newList
    #***********Inner Function*******************************
    #Replace all non-words characters with blanks, then split on blanks
    def tokenizerSplitter(string):
        string = string.upper()
        string = re.sub('[\W]+',' ',string)
        tokenList = re.split('[\s]+',string)
        newList = []
        for token in tokenList:
            if len(token)>0:
                newList.append(token)
        return newList
    #***********Inner Function*******************************
    #Replace delimiter with blanks, then compress token by replacing non-word characters with null
    def tokenizerCompressNbr(string):
        string = string.upper()
        print('=====tokenizerCompressNbr')
        print('string=',string)
        fieldList = re.split(delimiter,string)
        print('fieldList=',fieldList)
        newList = []
        for fieldValue in fieldList:
            print('fieldValue=',fieldValue)          
            tokenList = fieldValue.split()
            print('tokenList=',tokenList)
            nbr = False
            listLen = len(tokenList)
            print('listLen=',listLen)
            newToken =''
            for j in range(listLen):
                token = tokenList[j]
                print('token=', token)
                token = re.sub('[\W]+','',token) 
                print('token=',token)
                if token.isdigit():
                    nbr = True
                    newToken = newToken + token
                else:
                    if nbr:
                        newList.append(newToken)
                        newToken = ''
                        nbr = False
                    newList.append(token)
            if nbr:
                newList.append(newToken)
        print('newList=', newList)
        return newList
    #***********Outer Main Function*******************************
    # Start of Main Tokenizer Function
    logFile = DWM10_Parms.logFile
    print('\n>> Starting DWM14')
    print('\n>> Starting DWM14', file=logFile)
    inputFileName = DWM10_Parms.inputFileName
    refDict = {}
    print('Input Reference File Name =',inputFileName)
    print('Input Reference File Name =',inputFileName, file=logFile)
    hasHeader = DWM10_Parms.hasHeader
    print('Input File has Header Records =', hasHeader)
    print('Input File has Header Records =', hasHeader, file=logFile)
    delimiter = DWM10_Parms.delimiter
    print('Input File Delimiter =',delimiter)
    print('Input File Delimiter =',delimiter, file=logFile)
    tokenizerType = DWM10_Parms.tokenizerType
    print('Tokenizer Function Type =',tokenizerType)
    print('Tokenizer Function Type =',tokenizerType, file=logFile)
    removeDuplicateTokens = DWM10_Parms.removeDuplicateTokens
    print('Remove Duplicate Reference Tokens =',removeDuplicateTokens)
    print('Remove Duplicate Reference Tokens =',removeDuplicateTokens, file=logFile)
    goodType = False
    if tokenizerType=='Splitter':
        tokenizerFunction = tokenizerSplitter
        goodType = True
    if tokenizerType=='Compress':
        tokenizerFunction = tokenizerCompress
        goodType = True
    if tokenizerType=='CompressNbr':
        tokenizerFunction = tokenizerCompressNbr
        goodType = True        
    if goodType == False:
        print('**Error: Invalid Parameter value for tokenizerType ',tokenizerType)
        sys.exit()
    # Read input file and build reference dictionary (refDict)
    inputFile= open(inputFileName,'r')
    refCnt = 0
    # skip header record
    if hasHeader:
        line = inputFile.readline()
    line = inputFile.readline()
    tokenCnt = 0
    tokensOut = 0
    tokenFreqCount = []
    while line !='':
        refCnt +=1
        line = line.strip()
        # assume first token is the reference identifier (refID)
        firstDelimiter = line.find(delimiter)
        refID = line[0:firstDelimiter]
        body = line[firstDelimiter+1:]
        tokenList = tokenizerFunction(body)
        tokenCnt = tokenCnt + len(tokenList)
        if removeDuplicateTokens:
            tokenList = list(dict.fromkeys(tokenList))
        tokensOut = tokensOut + len(tokenList)
        refDict[refID] = tokenList          
        line = inputFile.readline()
    inputFile.close()
    print('Total References Read=',refCnt)
    print('Total References Read=',refCnt, file=logFile)
    print('Total Tokens Found =',tokenCnt)
    print('Total Tokens Found =',tokenCnt, file=logFile)
    return refDict

