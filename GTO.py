# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 17:57:39 2021

@author: elfares
@Email: ifares.cs@gmail.com
"""

import random
import numpy
import math
from solution import solution
import time
import transfer_functions_benchmark
import fitnessFUNs
import statistics

def GTO(objf,lower_bound,upper_bound,dim,SearchAgents_no,Max_iter,trainInput,trainOutput):



    #dim=30
    #SearchAgents_no=50
    #lb=-100
    #ub=100
    #Max_iter=500
    PopSize = 20
    
    fitness=numpy.zeros(PopSize)
    
    # initialize position vector and score for the leader
    ## Leader_pos=numpy.zeros(dim)
    ## Silverback_Score=float("inf")  #change this to -inf for maximization problems
    
    Silverback=numpy.zeros(dim)
    Silverback_Score=float("inf")  #change this to -inf for maximization problems
    
    
    #Initialize the positions of search agents
    #Positions=numpy.random.uniform(0,1,(SearchAgents_no,dim)) *(upper_bound-lower_bound)+lower_bound #generating continuous individuals
    Positions=numpy.random.randint(2, size=(SearchAgents_no,dim)) #generating binary individuals
    #Initialize convergence
    convergence_curve1=numpy.zeros(Max_iter)
    convergence_curve2=numpy.zeros(Max_iter)

    
    ############################
    s=solution()

    print("GTO is optimizing  \""+objf.__name__+"\"")    

    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    ############################
    
    
    for i in range(0,PopSize):
        while numpy.sum(Positions[i,:])<=0:   
            Positions[i,:]=numpy.random.randint(2, size=(1,dim))
        fitness[i]=objf(Positions[i,:],trainInput,trainOutput,dim)
        if fitness[i]< Silverback_Score:
            Silverback_Score = fitness[i]
            Silverback = Positions[i,:]
            
    
    GX = Positions
    lb = numpy.ones((1,dim), dtype=int)*lower_bound
    ub = numpy.ones((1,dim), dtype=int)*upper_bound
    
    
    ###  Controlling parameter
    p=0.03
    Beta=3
    w=0.8
    
    
    ### Main loop
    for It in range(0,Max_iter):
        a_Par=(math.cos(2*random.uniform(0, 1))+1)*(1-It/Max_iter)
        C=a_Par*(2*random.uniform(0, 1)-1)
        for i in range(0,PopSize):
            if random.uniform(0, 1)<p:
                GX[i,:] = (ub-lb)*random.uniform(0, 1)+lb
            else:
                if random.uniform(0, 1)>0.5:
                    Z = numpy.zeros(dim)
                    for j in range(0,dim):
                        Z[j]= random.uniform(- a_Par,a_Par)
                    H= Z * Positions[i,:]
                    GX[i,:]=(random.uniform(0, 1)-a_Par)*Positions[random.randint(0,PopSize-1),:]+C*H
                else:
                    GX[i,:]=Positions[i,:]-C*(C*(Positions[i,:]- GX[random.randint(0,PopSize-1),:])+random.uniform(0, 1)*(Positions[i,:]-GX[random.randint(0,PopSize-1),:]))
        
        #GX = BoundaryCheck(GX, lower_bound, upper_bound)
      
        
        
        
        ##  Group formation operation 
        for i in range(0,PopSize):
            GX[i,:]  = numpy.clip(GX[i,:], lb, ub)
            while numpy.sum(GX[i,:])<=0:   
                 GX[i,:]=numpy.random.randint(2, size=(1,dim))
            New_Fit= objf(GX[i,:],trainInput,trainOutput,dim)         
            if New_Fit<fitness[i]:
                fitness[i]=New_Fit
                Positions[i,:]=GX[i,:]
            if New_Fit<Silverback_Score:
                 Silverback_Score=New_Fit 
                 Silverback=GX[i,:] 
                        
                    
        ## Exploitation:   
        for i in range(0,PopSize):
            if a_Par>=w:
                g=2**C
              
                Mean =numpy.zeros(len(GX))
                for j in GX:
                    Mean[j]=abs(statistics.mean(j))
                delta= (Mean**g)**(1/g)
                GX[i,:]=C*delta[i]*(Positions[i,:]-Silverback)+Positions[i,:]
            else:
                if random.uniform(0, 1)>=0.5:
                    h=random.randint(0,PopSize-1)
                else:
                    h=random.randint(1,1)
                r1=random.uniform(0, 1) 
                GX[i,:]= Silverback-(Silverback*(2*r1-1)-Positions[i,:]*(2*r1-1))*(Beta*h); 
         
            
        #GX = BoundaryCheck(GX, lower_bound, upper_bound)
        #GX[i,:]  = numpy.clip(GX[i,:], lb, ub)
       
        
         ##  Group formation operation 
        for i in range(0,PopSize):
            GX[i,:]  = numpy.clip(GX[i,:], lb, ub)
            while numpy.sum(GX[i,:])<=0:   
                 GX[i,:]=numpy.random.randint(2, size=(1,dim))
            New_Fit= objf(GX[i,:],trainInput,trainOutput,dim)          
            if New_Fit<fitness[i]:
                fitness[i]=New_Fit
                Positions[i,:]=GX[i,:]
            if New_Fit<Silverback_Score:
                 Silverback_Score=New_Fit 
                 Silverback=GX[i,:] 
         
        featurecount=0
        for f in range(0,dim):
            if Silverback[f]==1:
                featurecount=featurecount+1
            
        convergence_curve1[It]=Silverback_Score
        convergence_curve2[It]=featurecount
        
        print(['At iteration '+ str(It+1)+ ' the best fitness on trainig is: '+ str(Silverback_Score)+'the best number of features: '+str(featurecount)])
        
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.bestIndividual=Silverback
    s.convergence1=convergence_curve1
    s.convergence2=convergence_curve2

    s.optimizer="GTO"
    s.objfname=objf.__name__
    
    return s


