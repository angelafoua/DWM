#!/usr/bin/env python
# coding: utf-8

# In[1]:


from textdistance import DamerauLevenshtein
import DWM10_Parms
def normalized_similarity(ref1, ref2):
    #print('--Starting DWM65')
    #print(ref1,'***',ref2)
    Class = DamerauLevenshtein()
    mu = DWM10_Parms.mu
    score = 0.0  
    m = len(ref1)
    n = len(ref2)
    if m==0 or n==0:
        return score
    #generate m x n matrix
    matrix = [[0.0 for j in range(n)] for i in range(m)]
    maxVal = -1.0
    #populate matrix with similarities between tokens
    for j in range(0,m):
        token1 = ref1[j]
        for k in range(0,n):
            token2 = ref2[k]
            simVal = 0.0
            #print('-Comparing',token1, token2)
            # Numeric Token Rule, if both tokens numeric, only exact match
            if DWM10_Parms.matrixNumTokenRule:
                if token1.isdigit() and token2.isdigit():
                    if token1==token2:
                        simVal = 1.0
                    else:
                        simVal = 0.0
                    #print('*Fired Rule 1', j, k, simVal)
                    matrix[j][k] = simVal                  
                    continue
            # Initial Rule, if either token length 1, only exact match
            if DWM10_Parms.matrixInitialRule:            
                if len(token1)==1 or len(token2)==1:
                    if token1==token2:
                        simVal = 1.0
                    else:
                        simVal = 0.0
                    #print('*Fired Rule 2',j, k, simVal)
                    matrix[j][k] = simVal             
                    continue
            # Default Rule, otherwise use Damerau-levesthein distance
            simVal = Class.normalized_similarity(token1,token2)
            #print('*Fired Rule 3', j, k, simVal)
            matrix[j][k] = simVal
    #end of matrix population       
    loops = 0 
    total = 0.0
    while True:
        maxVal = -1.0
        # search for maximum value in matrix
        for j in range(m):
            for k in range(n):
                if matrix[j][k]>maxVal:
                    maxVal = matrix[j][k]
                    saveJ = j
                    saveK = k
        #print('-*Max Value ', maxVal, ' found at ', saveJ, saveK)
        if maxVal < 0:
            #print('-Normal Ending no more postive values, loops =', loops, score)
            return score
        total = total + maxVal
        loops +=1
        score = total/loops
        if score < mu:
            #print('-Ending because score below mu =',loops, score)
            return score
        # set column saveK values to -1.0
        for j in range(m):
            matrix[j][saveK] = -1.0
        # set row saveJ values to -1.0
        for k in range(n):
            matrix[saveJ][k] = -1.0  
    #end of while loop
    #print('-Should never see this message',loops, score)
    return score

