#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import DWM10_Parms
def buildLinkIndex(refDict):
    logFile = DWM10_Parms.logFile
    print('\n>>Starting DWM15')
    print('\n>>Starting DWM15', file=logFile)
    linkIndex = {}
    for key in refDict:
        linkIndex[key] =''
    print('LinkIndex created, record count =', len(linkIndex))
    print('LinkIndex created, record count =', len(linkIndex), file=logFile)
    return linkIndex

