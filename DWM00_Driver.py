#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import time
import datetime
import logging
import DWM10_Parms
import DWM14_BuildRefDict
import DWM15_BuildLinkIndex
import DWM16_BuildTokenFreqDict
import DWM25_Global_Token_Replace
import DWM42_BuildBlockPairs
import DWM42_VectorBlockPairs
import DWM45_Block_Cleaning
import DWM55_LinkBlockPairs
import DWM80_TransitiveClosure
import DWM90_IterateClusters
import DWM96_WriteLinkIndex
import DWM97_ClusterProfile
import DWM99_ERmetrics
import DWM100_ReportData
import xlsxwriter


# In[2]:


# Main Driver for Refactored Data Washing Machine
# Version 1.20 creates a log file with same information being written to console
# Version 1.30 creates cluster profile at end of program and evaluates ER statistics
# Version 1.40 FK - added module DWM25 to do global level token replacement
#              JRT - added DWM65_ScoringMatrix to allow ScoringMatrix as a comparitor type
# Version 1.50 Revised and corrected scoring matrix
#              Revised DWM25 Global Replacement to reuse Tokenizer Dictionary and use DWM_WordList.txt
# Version 1.60 Implemented 2 versions of Scoring Rule - Standard (Std) and Weighted (Kris)
#              Changed Parms to be a class imported by all modules
# Version 1.70 Added new parameter minBlkTokenLen to set a minimum length for blokcing tokens
#              Improved performance of global cleaning
# Version 1.80 Added new parameter excludeNumericBlocks when True does not block on numeric tokens
#              Added new parameter removeExcludedBlkTokens when True removes tokens excluded by
#                 minBlkTokenLen & exludeNumbericBlocks
#              Added timer and added Total Runtime to logging statistics
# Version 1.90 Extensive refactor of the processing logic separating stop word removal from blocking.
#              Also generate and deduplicate pairs from all blocks before comparing or correcting references
#              Added a new parameter blockByPairs requiring refs in a block to share 2 tokens
# Version 2.00 Added new parameter runIterationProfile to print profile & stats at end of each iteration
#              Also generated new token statistics for number of all-digit (numeric) tokens and ratio to total
#              Added token length profile and statistics
# Version 2.10 Added DWM100 for reporting to create a spreadsheet of statistics with parameter settings.
#              Added DWM45 with a new parameter blockCorrection to correct blocking tokens.
#              Added changes to DWM25 and DWM80 for faster processing of data.
# Version 2.20 Both global and block corrections output to log file instead of separate files
#              3 new parms, globalCorrectionDetail, blockCorrectionDetail, addRefsToLinkIndex
# Version 2.21 Corrected two bugs, one in Global Correction, and one in DWM90
# Version 2.30 Added new value 'CompressNbr' to parameter 'tokenizerType' 
# Version 2.31 Corrected a bug in DWM42 where last key group was not being processed
version = 2.31

# get start time for timer
startTime = time.time()
# date time is used to label the logfile
now = datetime.datetime.now()
tag = str(now.year)+(str(now.month)).zfill(2)+(str(now.day)).zfill(2)
tag = tag+'_'+(str(now.hour)).zfill(2)+'_'+(str(now.minute)).zfill(2)
logFile = open('DWM_Log_'+tag+'.txt','w')
print("Data Washing Machine Refactor Version",version)
print("Data Washing Machine Refactor Version",version, file=logFile)
print("Date/Time",tag)
print("Data/Time",tag, file=logFile)
while True:
    choice = input('Enter 1 to run single parms file, Enter 2 to run a list of parms files ->')
    if choice == '1':
        multi = False
        parmFileName = input('Enter Name of a Single Parameter File ->')
        break
    if choice == '2':
        multi = True
        fileName = input('Enter Name of a List of Parameter Files ->')
        file1 = open(fileName, 'r')
        break
    print('Try again')
#data reporting init
excelFileName = 'DWM_Results_'+tag+'.xlsx'
DWM10_Parms.workbook = xlsxwriter.Workbook(excelFileName)
DWM10_Parms.worksheet = DWM10_Parms.workbook.add_worksheet()
DWM10_Parms.startRow = 0
while True:    
    now1 = datetime.datetime.now()
    if multi == True:
        parmFileName = file1.readline()
        print('\n\nRunning parms file',parmFileName)
        print('\nRunning parms file ',parmFileName, file=logFile)
        parmFileName = parmFileName.replace('\n','')
        if not parmFileName:
            print('\nEnd of the parmFileName Runs')
            break
    else:
        print('\n\nRunning parms file',parmFileName)
        print('\nRunning parms file ',parmFileName, file=logFile)   
    DWM10_Parms.getParms(parmFileName, logFile)
    # Must get mu start and save to muStart for single parms file
    # and epsilonStart for parmeter single parms file
    # DWM10_Parms.blockCorrection changes if DWM45 runs need original value for reporting
    DWM10_Parms.muStart=DWM10_Parms.mu
    DWM10_Parms.epsilonStart=DWM10_Parms.epsilon
    DWM10_Parms.blockCorrect =DWM10_Parms.blockCorrection
    # Create refDict, a dictionary where key=refID, value is list of reference tokens
    refDict = DWM14_BuildRefDict.tokenizeInput()
    # Create linkIndx, a dictionary where key=refID, value is cluster ID`
    linkIndex = DWM15_BuildLinkIndex.buildLinkIndex(refDict)
    # Create tokenFeqDict, a dictionary where key=token, value is token frequency
    tokenFreqDict =DWM16_BuildTokenFreqDict.buildTokenFreqDict(refDict)
    # create dictionary of corrections (stdTokenDict), leave empty if not running replacement
    # Global correction intentionally disabled; log when requested.
    if DWM10_Parms.runGlobalCorrection:
        print('runGlobalCorrection configured; skipping globalReplace in DWM00_Driver')
        print('runGlobalCorrection configured; skipping globalReplace in DWM00_Driver', file=logFile)
    moreToDo = True
    print('\n>>Starting Iterations')
    print('\n>>Starting Iterations', file=logFile)
    mu = DWM10_Parms.mu
    print('mu start value=', mu)
    print('mu start value=', mu, file=logFile)
    muIterate = DWM10_Parms.muIterate
    print('mu iterate value=', muIterate)
    print('mu iterate value=', muIterate, file=logFile)
    epsilon = DWM10_Parms.epsilon
    print('epsilon start value=', epsilon)
    print('epsilon start value=', epsilon, file=logFile)
    epsilonIterate = DWM10_Parms.epsilonIterate
    print('epsilon iterate value=', epsilonIterate)
    print('epsilon iterate value=', epsilonIterate, file=logFile)
    comparator = DWM10_Parms.comparator
    print('comparator =', comparator)
    print('comparator =', comparator, file=logFile)
    firstIteration = True
    while moreToDo:
        print('\n****New Iteration\nSize of refDict =', len(refDict))   
        print('\n****New Iteration\nSize of refDict =', len(refDict), file=logFile)  
        #blockList = DWM40_BuildBlocks.buildBlocks(logFile, refList, tokenFreqDict)
        if DWM10_Parms.useVectorBlocking:
            blockPairList = DWM42_VectorBlockPairs.buildVectorBlockPairs(refDict, linkIndex, tokenFreqDict)
        else:
            blockPairList = DWM42_BuildBlockPairs.buildBlockPairs(refDict, linkIndex, tokenFreqDict)
        if len(blockPairList)==0:
            print('--Ending because blockPairList is empty')
            print('--Ending because blockPairList is empty', file=logFile)
            break
        # If block correction requested, only run once on first iteration
        if DWM10_Parms.blockCorrection and firstIteration:
            changeCount = DWM45_Block_Cleaning.RunBlockCorrections(blockPairList, tokenFreqDict, refDict)
            # if there were block corrections, rebuild token dictionary and re-block
            if changeCount > 0:
                tokenFreqDict=DWM16_BuildTokenFreqDict.buildTokenFreqDict(refDict)
                if DWM10_Parms.useVectorBlocking:
                    blockPairList = DWM42_VectorBlockPairs.buildVectorBlockPairs(refDict, linkIndex, tokenFreqDict)
                else:
                    blockPairList = DWM42_BuildBlockPairs.buildBlockPairs(refDict, linkIndex, tokenFreqDict)                
            firstIteration = False
        linkedPairList = DWM55_LinkBlockPairs.linkBlockPairs(blockPairList, refDict, tokenFreqDict)
        if len(linkedPairList)==0:
            print('Ending because linkedPairList is empty')
            print('Ending because linkedPairList is empty', file=logFile)
            break
        clusterList = DWM80_TransitiveClosure.transitiveClosure(linkedPairList)
        if len(clusterList)==0:
            print('--Ending because clusterList is empty') 
            print('--Ending because clusterList is empty', file=logFile)
            break  
        iterationLinkIndex = DWM90_IterateClusters.iterateClusters(clusterList, refDict, linkIndex)
        print("\n>>Itermediate Results from this Iteration")
        print("\n>>Itermediate Results from this Iteration", file=logFile)
        # Run iteration profile and statistics if requested
        if DWM10_Parms.runIterationProfile:
            DWM97_ClusterProfile.generateProfile(iterationLinkIndex)
            if DWM10_Parms.truthFileName != '':
                DWM99_ERmetrics.generateMetrics(iterationLinkIndex)
        print('\n>>End of Iteration, Resetting mu and epsilon')
        print('\n>>End of Iteration, Resetting mu and epsilon', file=logFile)
        mu += muIterate
        mu = round(mu, 2)
        DWM10_Parms.mu = mu
        print('>>>New Value of mu = ',mu)
        print('>>>New Value of mu = ',mu, file=logFile)
        epsilon += epsilonIterate
        epsilon = round(epsilon, 2)
        DWM10_Parms.epsilon = epsilon
        print('>>>New Value of epsilon = ',epsilon)
        print('>>>New Value of epsilon = ',epsilon, file=logFile)
        if mu > 1.0:
            moreToDo = False
            print('Ending because mu > 1.0')
            print('Ending because mu > 1.0', file=logFile)
    # End of iterations
    # write Link Index to text file
    DWM96_WriteLinkIndex.writeLinkIndex(linkIndex, refDict)
    # Generate Cluster Profile
    DWM97_ClusterProfile.generateProfile(linkIndex)
    # Generat ER Metrics if truthFileName was given
    if DWM10_Parms.truthFileName != '':
        DWM99_ERmetrics.generateMetrics(linkIndex)
        DWM100_ReportData.reportData()
    now2 = datetime.datetime.now()
    print("\nTotal File Runtime =", now2-now1, file=logFile)
    print("\nEnd of File ",parmFileName)
    print('Time to run File ', now2-now1)
    print("End of File ",parmFileName, file=logFile)
    if multi==False:
        break
if multi==True:
    file1.close()
endTime = time.time()
totalTime = endTime - startTime
print("All Files Total Runtime =", totalTime/60, " minutes")
print("All Files Total Runtime =", totalTime/60, " minutes", file=logFile)
print("End of Program")
print("End of Program", file=logFile)
logFile.close()
DWM10_Parms.workbook.close()

