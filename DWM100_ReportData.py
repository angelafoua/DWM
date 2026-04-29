#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xlsxwriter
from csv import reader
import DWM10_Parms

def reportData():
 
    # Get the workbook from DWM10 .
    workbook = DWM10_Parms.workbook
    #load the captured data from DWM10
    dataList = loadData()
    #add formatting into workbook for use by worksheet
    cell_format = workbook.add_format({'bold': True,'valign': 'vcenter','align':'center'})
    cell_format.set_font_name('Arial Narrow')
    cell_format.set_bg_color('#808080')
    cell_format.set_font_color('white')
    cell_format2 = workbook.add_format({'align':'left'})
    center = workbook.add_format({'align': 'center','valign':'vcenter'})
    
    # need to know what startrow is each time - changes in Multi-Driver each time
    # and stored in DWM10
    startRow = DWM10_Parms.startRow
    
    if startRow == 0:
        
        #Create Header Row only once startRow ==0
        worksheet = DWM10_Parms.worksheet
        
        # open file in read mode - do not need to close when using 'with'
        with open('DWMDataCaptureHeader.csv', 'r') as read_obj:
            csv_reader = reader(read_obj)
            string = []
            for row in csv_reader:
                string = row
                #only one row
                break
               
        i = 0        
        # walk through the values and add to correct column
        #this method makes sure changes to the columns are picked up
        length = len(string)
        wide = 0
        while i < length:
            # pad the headers for easier reading
            leng = len(string[i].strip())
            if leng >8:
                leng = leng + 7
            else:
                leng = leng + 4
            
            #how wide so we can freeze the top row
            wide+=leng
            #set the column widths                
            worksheet.set_column(i, i, leng)            
            worksheet.write(0,i,string[i].upper().strip(),cell_format)
            i += 1
       
        #freeze header height by width        
        worksheet.split_panes(1,wide)
        #freeze first 4 columns
        worksheet.freeze_panes(1, 4)
        
        # now write the first dataList to first row under header   
        worksheet.write_row(1, 0, dataList,center)
        DWM10_Parms.dataList = []
        #increment counter for first row startRow==0 here
        DWM10_Parms.startRow+=1        
      
    else:
        # this is why need worksheet in parms or could not initialize it
        worksheet = DWM10_Parms.worksheet
        # write all the rest of records as they come in here once header is setup
        worksheet.write_row(startRow, 0, dataList,center)
        DWM10_Parms.dataList = []
    #increment counter   
    DWM10_Parms.startRow+=1
    
    return

def loadData():
    
    dataList = DWM10_Parms.dataList 
    dataList.append(DWM10_Parms.inputFileName)
    dataList.append(DWM10_Parms.precision)
    dataList.append(DWM10_Parms.recall)
    dataList.append(DWM10_Parms.fmeasure)
    dataList.append(DWM10_Parms.linkedPairs)
    dataList.append(DWM10_Parms.expectedPairs)
    dataList.append(DWM10_Parms.truePairs)
    dataList.append(DWM10_Parms.tokenizerType) # add this
    dataList.append(DWM10_Parms.refCnt)
    dataList.append(DWM10_Parms.tokenCnt)
    dataList.append(DWM10_Parms.uniqueTokenCnt)
    dataList.append(DWM10_Parms.uniqueTokenRatio)
    dataList.append(DWM10_Parms.numTokenCnt)
    dataList.append(DWM10_Parms.numTokenRatio)
    dataList.append(DWM10_Parms.minFreq)
    dataList.append(DWM10_Parms.maxFreq)
    dataList.append(DWM10_Parms.avgFreq)
    dataList.append(DWM10_Parms.stdFreq)
    dataList.append(DWM10_Parms.minLen)
    dataList.append(DWM10_Parms.maxLen)    
    dataList.append(DWM10_Parms.avgLen)
    dataList.append(DWM10_Parms.stdDevLen)
    dataList.append(DWM10_Parms.runGlobalCorrection)
    dataList.append(DWM10_Parms.minFreqStdToken)
    dataList.append(DWM10_Parms.minLenStdToken)
    dataList.append(DWM10_Parms.maxFreqErrToken)
    dataList.append(DWM10_Parms.beta)
    dataList.append(DWM10_Parms.blockByPairs)   
    dataList.append(DWM10_Parms.minBlkTokenLen)  #spelling error in spreadsheet
    dataList.append(DWM10_Parms.excludeNumericBlocks) #spelling error in spreadsheet
    dataList.append(DWM10_Parms.sigma)
    dataList.append(DWM10_Parms.removeDuplicateTokens)
    dataList.append(DWM10_Parms.removeExcludedBlkTokens) #spelling error in spreadsheet    
    # Single parms files need to list the original start value in else:
    if DWM10_Parms.runIterationProfile==True:
        dataList.append(DWM10_Parms.epsilon)
        dataList.append(DWM10_Parms.epsilonIterate)
        dataList.append(DWM10_Parms.mu)        
    else:
        dataList.append(DWM10_Parms.epsilonStart)
        dataList.append(DWM10_Parms.epsilonIterate)
        dataList.append(DWM10_Parms.muStart)    
    dataList.append(DWM10_Parms.muIterate)
    dataList.append(DWM10_Parms.comparator)
    dataList.append(DWM10_Parms.matrixNumTokenRule)
    dataList.append(DWM10_Parms.matrixInitialRule)
    # Report the run start value and not after the value is changed
    # in the blockCorrection process
    dataList.append(DWM10_Parms.blockCorrect)    
  
    return dataList

