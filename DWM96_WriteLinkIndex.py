#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import DWM10_Parms
def writeLinkIndex(linkIndex, refDict):
    logFile = DWM10_Parms.logFile
    print('\n>>Starting DWM96')
    print('\n>>Starting DWM96', file=logFile)    
    # add any missing sigleton clusters
    for key in linkIndex:
        if len(linkIndex[key])==0:
            linkIndex[key] = key
    # write out linkIndex
    inputPrefix = DWM10_Parms.inputPrefix
    linkFileName = inputPrefix+'-LinkIndex.txt'
    linkFile = open(linkFileName,'w')
    linkFile.write('RefID, ClusterID\n')
    for pair in sorted(linkIndex.items()):
        refID = pair[0]
        clusterID = pair[1]
        head = refID + ', ' + clusterID
        if DWM10_Parms.addRefsToLinkIndex:
            tokenList = refDict[refID]
            head = head+','
            for token in tokenList:
                head = head + ' ' + token
        linkFile.write(head+'\n')
    linkFile.close()
    print('Record written to',linkFileName, '=',len(linkIndex))
    print('Record written to',linkFileName, '=',len(linkIndex), file=logFile)
    return

