#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
#coding: utf-8

import sys
#####################################
# Parameters set by the User Script
#####################################
# Input Parameters
inputFileName = ''
delimiter = ','
hasHeader = False
tokenizerType = 'Splitter'
truthFileName = ''
runIterationProfile = False
addRefsToLinkIndex = False
# Global Correction Parameters
runGlobalCorrection = False
globalCorrectionDetail = False
minFreqStdToken = 5
minLenStdToken = 3
maxFreqErrToken = 3
# Blocking Parameters
beta = 2
blockByPairs = True
minBlkTokenLen = 4
excludeNumericBlocks = True
# Vector blocking parameters (used by DWM42_VectorBlockPairs)
useVectorBlocking = False       # True → use ANN/vector blocking; False → classic DWM42
vectorBlockThreshold = 0.5      # cosine similarity threshold for candidate pairs
vectorBlockTopK = 10            # number of ANN neighbours to retrieve per record
# Block Correction Parameters
blockCorrection = False
blockCorrectionDetail = False
# Linking Parameters
epsilon = 0.50
epsilonIterate = 0.00
mu = 0.50
muIterate = 0.10
comparator = 'ScoringMatrixStd'
matrixNumTokenRule = False
matrixInitialRule = False
# Stop Word Parameters
sigma = 12
removeDuplicateTokens = False
removeExcludedBlkTokens = True
##################################################
# Internal Parameters set by the program
##################################################
inputPrefix = ''
logFile = ''
muStart = 0.00
epsilonStart = 0.00
fatalError = False
workbook = None
worksheet = None
startRow = 0
dataList = []
# Run Statistics
refCnt = 0
tokenCnt = 0
uniqueTokenRatio = 0.00
numTokenCnt = 0
numTokenRatio = 0.00
minFreq = 0
maxFreq = 0
avgFreq = 0.00
stdFreq = 0.00
minLen = 0
maxLen = 0
avgLen = 0.00
stdDevLen = 0.00
precision = 0.00
recall = 0.00
fMeasure = 0.00
truePairs = 0
linkedPairs = 0
expectedPairs = 0
###########################################
# Helper Functions
###########################################
def convertToBoolean(lineNbr, value):
    if value=='True':
        return True
    if value=='False':
        return False
    print('**Error: Invalid Boolean value in Parameter File, line:',lineNbr,'->',value)
    global fatalError
    fatalError = True
def convertToFloat(lineNbr, value):
    try:
        floatValue = float(value)
    except ValueError:
        print('**Error: Invalid floating point value in Parameter File, line:',lineNbr,'->',value)
        global fatalError
        fatalError = True
    else:
        return floatValue
def convertToInteger(lineNbr, value):
    if value.isdigit():
        return int(value)
    else:
        print('**Error: Invalid integer value in Parameter File, line:',lineNbr,'->',value)
        global fatalError
        fatalError = True
###############################################
# Main Program
###############################################
def getParms(parmFileName, logName):
    global logFile
    logFile = logName
    global fatalError
   
    validParmNames = ['inputFileName','delimiter', 'hasHeader', 'tokenizerType', 'removeDuplicateTokens',                        'minFreqStdToken', 'minLenStdToken', 'maxFreqErrToken', 'addRefsToLinkIndex',                              'mu', 'muIterate', 'beta', 'minBlkTokenLen', 'sigma', 'epsilon', 'epsilonIterate',                         'excludeNumericBlocks', 'removeExcludedBlkTokens','runClusterMetrics', 'createFinalJoin',                       'blockByPairs', 'comparator','truthFileName', 'matrixNumTokenRule', 'matrixInitialRule',                        'runGlobalCorrection', 'runIterationProfile', 'blockCorrection', 'blockCorrectionDetail',                       'globalCorrectionDetail',                        'useVectorBlocking', 'vectorBlockThreshold', 'vectorBlockTopK']
    parmFile = open(parmFileName,'r')
    parms = {}
    lineNbr = 0
    while True:
        line = (parmFile.readline()).strip()
        lineNbr +=1
        if line=='':
            break
        # Skip comment lines in parameter file
        if  line.startswith('#'):
            continue
        if line.find('=') < 0:
            print('**Error: Parameter line does not have equal sign, line:',lineNbr,'->',line)
            fatalError = True
            continue
        part = line.split('=')
        parmName = part[0].strip()
        if parmName not in validParmNames:
            print('**Error: Invalid Parameter Name in Parameter File, line:',lineNbr,'->',parmName)
            fatalError = True
        parmValue = part[1].strip()
        if parmName=='inputFileName':
            global inputFileName
            inputFileName = parmValue
            periodIndex = inputFileName.rfind('.')
            global inputPrefix
            inputPrefix = inputFileName[0:periodIndex]
            continue
        if parmName=='delimiter':
            global delimiter
            if ',;:|\t'.find(parmValue)>=0:
                delimiter = parmValue
                continue
            else:
                print('**Error: Invalid delimiter in Parameter File, line:',lineNbr,'->',parmName)
                sys.exit()                             
        if parmName=='hasHeader':
            global hasHeader
            hasHeader = convertToBoolean(lineNbr, parmValue)
            continue    
        if parmName=='tokenizerType':
            global tokenizerType
            tokenizerType = parmValue
            continue
        if parmName=='removeDuplicateTokens':
            global removeDuplicateTokens
            removeDuplicateTokens = convertToBoolean(lineNbr, parmValue)
            continue
        if parmName=='runGlobalCorrection':
            global runGlobalCorrection
            runGlobalCorrection = convertToBoolean(lineNbr, parmValue)
            continue
        if parmName=='globalCorrectionDetail':
            global globalCorrectionDetail
            globalCorrectionDetail = convertToBoolean(lineNbr, parmValue)
            continue        
        if parmName=='runIterationProfile':
            global runIterationProfile
            runIterationProfile = convertToBoolean(lineNbr, parmValue)
            continue
        if parmName=='addRefsToLinkIndex':
            global addRefsToLinkIndex
            addRefsToLinkIndex = convertToBoolean(lineNbr, parmValue)
            continue            
        if parmName=='minFreqStdToken':
            global minFreqStdToken
            minFreqStdToken = convertToInteger(lineNbr, parmValue)
            continue
        if parmName=='minLenStdToken':
            global minLenStdToken
            minLenStdToken = convertToInteger(lineNbr, parmValue)
            continue
        if parmName=='maxFreqErrToken':
            global maxFreqErrToken
            maxFreqErrToken = convertToInteger(lineNbr, parmValue)
            continue            
        if parmName=='matrixNumTokenRule':
            global matrixNumTokenRule
            matrixNumTokenRule = convertToBoolean(lineNbr, parmValue)
            continue
        if parmName=='matrixInitialRule':
            global matrixInitialRule
            matrixInitialRule = convertToBoolean(lineNbr, parmValue)
            continue
        if parmName=='mu':
            global mu
            mu = convertToFloat(lineNbr, parmValue)
            muStart = mu
            continue            
        if parmName=='muIterate':
            global muIterate
            muIterate = convertToFloat(lineNbr, parmValue)
            continue
        if parmName=='epsilon':
            global epsilon
            epsilon = convertToFloat(lineNbr, parmValue)
            epsilonStart = epsilon
            continue
        if parmName=='epsilonIterate':
            global epsilonIterate
            epsilonIterate = convertToFloat(lineNbr, parmValue)
            continue
        if parmName=='comparator':
            global comparator
            comparator = parmValue
            continue  
        if parmName=='beta':
            global beta
            beta = convertToInteger(lineNbr, parmValue)
            continue
        if parmName=='minBlkTokenLen':
            global minBlkTokenLen
            minBlkTokenLen = convertToInteger(lineNbr, parmValue)
            continue            
        if parmName=='excludeNumericBlocks':
            global excludeNumericBlocks
            excludeNumericBlocks = convertToBoolean(lineNbr, parmValue)
            continue            
        if parmName=='blockByPairs':
            global blockByPairs
            blockByPairs = convertToBoolean(lineNbr, parmValue)
            continue
        if parmName=='useVectorBlocking':
            global useVectorBlocking
            useVectorBlocking = convertToBoolean(lineNbr, parmValue)
            continue
        if parmName=='vectorBlockThreshold':
            global vectorBlockThreshold
            vectorBlockThreshold = convertToFloat(lineNbr, parmValue)
            continue
        if parmName=='vectorBlockTopK':
            global vectorBlockTopK
            vectorBlockTopK = convertToInteger(lineNbr, parmValue)
            continue
        if parmName=='removeExcludedBlkTokens':
            global removeExcludedBlkTokens
            removeExcludedBlkTokens = convertToBoolean(lineNbr, parmValue)
            continue            
        if parmName=='sigma':
            global sigma
            sigma = convertToInteger(lineNbr, parmValue)
            continue
        if parmName=='truthFileName':
            global truthFileName
            truthFileName = parmValue
            continue
        if parmName=='blockCorrection':
            global blockCorrection
            blockCorrection = convertToBoolean(lineNbr,parmValue)
            continue
        if parmName=='blockCorrectionDetail':
            global blockCorrectionDetail
            blockCorrectionDetail = convertToBoolean(lineNbr,parmValue)
            continue       
        if parmName=='workbook':
            global workbook
            workbook = parmValue
            continue
        if parmName=='worksheet':
            global worksheet
            worksheet = parmValue
            continue              
        if parmName=='startRow':
            global startRow
            startRow =  convertToInteger(lineNbr, parmValue)
            continue
        if parmName=='dataList':
            global dataList
            dataList = parmValue
            continue         
        if parmName=='refCnt':
            global refCnt
            refCnt =  convertToInteger(lineNbr, parmValue)
            continue
        if parmName=='tokenCnt':
            global tokenCnt
            tokenCnt =  convertToInteger(lineNbr, parmValue)
            continue
        if parmName=='uniqueTokenRatio':
            global uniqueTokenRatio
            uniqueTokenRatio = convertToFloat(lineNbr, parmValue)
            continue
        if parmName=='numTokenCnt':
            global numTokenCnt
            numTokenCnt =  convertToInteger(lineNbr, parmValue)
            continue
        if parmName=='numTokenRatio':
            global numTokenRatio
            numTokenRatio = convertToFloat(lineNbr, parmValue)
            continue
        if parmName=='minFreq':
            global minFreq
            minFreq =  convertToInteger(lineNbr, parmValue)
            continue
        if parmName=='maxFreq':
            global maxFreq
            maxFreq =  convertToInteger(lineNbr, parmValue)
            continue
        if parmName=='avgFreq':
            global avgFreq
            avgFreq = convertToFloat(lineNbr, parmValue)
            continue
        if parmName=='stdFreq':
            global stdFreq
            stdFreq = convertToFloat(lineNbr, parmValue)
            continue
        if parmName=='avgLen':
            global avgLen
            avgLen = convertToFloat(lineNbr, parmValue)
            continue
        if parmName=='stdDevLen':
            global stdDevLen
            stdDevLen = convertToFloat(lineNbr, parmValue)
            continue
        if parmName=='minLen':
            global minLen
            minLen =  convertToInteger(lineNbr, parmValue)
            continue
        if parmName=='maxLen':
            global maxLen
            maxLen =  convertToInteger(lineNbr, parmValue)
            continue
        if parmName=='precision':
            global precision
            precision = convertToFloat(lineNbr, parmValue)
            continue
        if parmName=='recall':
            global recall
            recall = convertToFloat(lineNbr, parmValue)
            continue
        if parmName=='fMeasure':
            global fMeasure
            fMeasure = convertToFloat(lineNbr, parmValue)
            continue
        if parmName=='truePairs':
            global truePairs
            truePairs =  convertToInteger(lineNbr, parmValue)
            continue
        if parmName=='linkedPairs':
            global linkedPairs
            linkedPairs =  convertToInteger(lineNbr, parmValue)
            continue
        if parmName=='expectedPairs':
            global expectedPairs
            expectedPairs =  convertToInteger(lineNbr, parmValue)
            continue
    ###############################################
    # End of Script cross checks
    if beta<2:
        print('**Error: beta value ', beta,' must be larger than 2')
        fatalError = True
    if sigma <= beta:
        print('**Error: sigma value ', sigma,' must be larger than beta value ', beta)
        fatalError = True
    if mu <= 0.0 or mu > 1.00:
        print('**Error: mu value ', mu,' must be in interval (0.00,1.00]')
        fatalError = True
    if muIterate < 0.0 or muIterate > 1.00:
        print('**Error: muIterate value ', muIterate,' must be in interval (0.00,1.00]')
        fatalError = True
    if epsilon <= 0.0 or epsilon > 1.00:
        print('**Error: epsilon value ', epsilon,' must be in interval (0.00,1.00]')
        fatalError = True
    if epsilonIterate < 0.0 or epsilonIterate > 1.00:
        print('**Error: epsilonIterate value ', epsilonIterate,' must be in interval (0.00,1.00]')
        fatalError = True
    if minFreqStdToken <= maxFreqErrToken:
        print('**Error: minFreqStdToken ', minFreqStdToken,' must be greater than maxFreqErrToken', maxFreqErrToken)
        fatalError = True
    if fatalError:
        sys.exit()  
    return

