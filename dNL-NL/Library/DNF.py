# Copyright 2019 Ali Payani.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import random
import collections
from time import sleep
from datetime import datetime
import sys
#import tensorflow as tf
import tensorflow.compat.v1 as tf
from itertools import product
from itertools import combinations_with_replacement
from .PredicateLibV5 import PredFunc

from .mylibw import *

class DNF(PredFunc):
    '''
    Primary logic layer for deriving predicates. Disjunctive Normal Form (DNF)
    We call a complex networks made by combining the elementary conjunctive and disjunctive neurons,
    a dNL (differentiable Neural Logic) network. For example, by cascading a conjunction layer with one
    disjunctive neuron we can form a dNL-DNF construct.
    conjunction layer: neural conjunction
    disjuntive layer: nerual disjuntion

    '''
    def __init__(self,name='',trainable=True,terms=4,sig=1.0,init=[.1,-1,.1,.1],init_terms=[],post_terms=[],predColl=None,off_w=-10):
        '''
        name: name of the predicate we're trying to define; 'class_1'
        trainable: boolean; true
        terms: ; 2
        sig: ; 2
        init: used to set the mean and std for the OR preds and AND preds; [-1, 0.1, -1, 0.1]
        init_terms:
        post_terms:
        predColl: None
        off_w: ; -10
        mean_or, std_or, mean_and, std_and: 
        col: list of [tensorflow.GraphKeys, string]; 
        sig: ; 2
        predColl: ; None
        '''
        
        super().__init__(name,trainable)
        self.terms = terms # 2
        self.init_terms = init_terms
        self.post_terms = post_terms
        self.mean_or,self.std_or,self.mean_and,self.std_and = init
        self.col = [tf.GraphKeys.GLOBAL_VARIABLES,self.name]
        self.sig = sig
        self.predColl=predColl
        self.off_w=off_w

    def pred_func(self,xi,xcs=None,t=0):
        '''
        This is equivalent to F_p^t, as in its the fuzzy predicate function
        used to learn the rules for our target classes
        xi: predicate matrix for continous inputs X1, x2, sqrX1, sqrX2, and corresponding lower/upper bound weights
        xcs: continous input tuple, NEVER GETS USED!!
        '''
        wa=None
        wo=None
        #this control statement is not relevant to our continuous pred learning
        if len(self.init_terms)>0:
            
            wa = np.zeros((self.terms,self.predColl[self.name].Lx),dtype=np.float32) 
            wo = np.zeros((1,self.terms)) 
            for i,a in enumerate(self.init_terms):
                
                for item in a.split(', '):
                    wa[i,self.predColl[self.name].get_term_index(item)]=10
                    wo[0,i]=10
            
            wa[wa<1] =self.off_w
            wo[wo<1] =self.off_w
            

            
        #xi would include the threshold values for the continuous predicates
        #xi (40, 112)

        #for OBS dNL xi~(40,56)
        temp = logic_layer_and( xi, self.terms ,  col=self.col, name=self.name+'_AND', sig=self.sig, mean=self.mean_and,std=self.std_and,w_init=wa)
        res = logic_layer_or( temp, 1 ,  col=self.col, name=self.name+'_OR',  sig=self.sig, mean=self.mean_or,std=self.std_or,w_init=wo ) 

        for t in self.post_terms:
            ind = self.predColl[self.name].get_term_index(t[1])
            if t[0]=='and':
                res = res * xi[:,ind:(ind+1)]
            if t[0]=='or':
                res = 1.0- (1.0-res) * (1.0-xi[:,ind:(ind+1)] )
                # res=logic_layer_or( (res,xi[:,ind:(ind+1)]), 1 ,  col=self.col, name=self.name+'_xx',  sig=self.sig, mean=self.mean_or,std=self.std_or ) 
        return res
    
    def conv_weight_np(self,w):
        return sharp_sigmoid_np(w,self.sig)
    def conv_weight(self,w):
        return sharp_sigmoid(w,self.sig)
    def get_func(self,session,names,threshold=.2,print_th=True):
        '''
        primary function for printing methods
        '''

        wt = tf.get_collection(self.name)
        
        if len(wt)<2:
            return ''
        
        if '_AND' in wt[0].name:
            w_andt = wt[0]
            w_ort  = wt[1]
        else:
            w_andt = wt[1]
            w_ort  = wt[0]

        w_and,w_or = session.run( [w_andt,w_ort] )
        w_and = sharp_sigmoid_np(w_and,self.sig) #silf.sig=2
        w_or = sharp_sigmoid_np(w_or,self.sig)
    

        clauses = []

        for k in range(w_or[0,:].size):
            if w_or[0,k]>threshold:
                
                terms=[]
                for v in range( w_and[k,:].size):
                    if w_and[k,v]>threshold:
                        if names is None:
                            terms.append( 'I_%d'%(v+1))
                        else:
                            terms.append( names[v])

                        if print_th and w_and[k,v]<.95:
                                terms[-1] = '[%.2f]'%(w_and[k,v]) + terms[-1]

                s = ','.join(terms)
                if print_th and w_or[0,k]<.9999:
                    clauses.append( '\t :- [%.2f] ('%(w_or[0,k]) + s +' )')
                else:
                    clauses.append( '\t :- ('%(w_or[0,k]) + s +' )')
        return '\n'.join(clauses)


    def getPredWeights(self, session, intervals = 2):
        '''
        get the weights for each target class 
        session: tensor session current state model
        '''
        wt = tf.get_collection(self.name)
        #print(wt)
        if len(wt)<2:
            return ''
        
        if '_AND' in wt[0].name:
            w_andt = wt[0]
            w_ort  = wt[1]
        else:
            w_andt = wt[1]
            w_ort  = wt[0]
        w_and,w_or = session.run( [w_andt,w_ort] )
        w_and = sharp_sigmoid_np(w_and,self.sig)
        w_or = sharp_sigmoid_np(w_or,self.sig)
        predcount = w_and.shape[1]
        preddefinitions = w_and.shape[0]
        pred_max_weights = []

        for d in range(preddefinitions):
            class_def_weights = []
            for k in range(0,predcount,2*intervals):
                #print(k)
                temp = w_and[d,k:k+2*intervals]
                max_weight = max(temp)
                class_def_weights.append(max_weight)
            pred_max_weights.append(class_def_weights)
        return pred_max_weights

    def getTensorWeights(self, intervals = 2):
        '''
        get the weights for each target class  based on the unprocessed tensor
        this method is responsible for collection the weights for each class tensor
        and using the intervals to find the maximum weights (ie predicates which are likely to be in the body of the class)
        
        The method is used in the defining our loss function, here we try to find the largest weight for a predicate interval
        as the weights correspond to the following predicates withe the sinX1 + expX2 example:

        ['X1', 'X1', 'X1', 'X1', 'X1', 'X1', 'X1', 'X1', 'X1Square', 'X1Square', 'X1Square', 'X1Square', 'X1Square', 'X1Square', 'X1Square', 'X1Square', 
        'X1Sine', 'X1Sine', 'X1Sine', 'X1Sine', 'X1Sine', 'X1Sine', 'X1Sine', 'X1Sine', 'X1Exp', 'X1Exp', 'X1Exp', 'X1Exp', 'X1Exp', 'X1Exp', 'X1Exp', 'X1Exp', 
        'X2', 'X2', 'X2', 'X2', 'X2', 'X2', 'X2', 'X2', 'X2Square', 'X2Square', 'X2Square', 'X2Square', 'X2Square', 'X2Square', 'X2Square', 'X2Square',
         'X2Sine', 'X2Sine', 'X2Sine', 'X2Sine', 'X2Sine', 'X2Sine', 'X2Sine', 'X2Sine', 'X2Exp', 'X2Exp', 'X2Exp', 'X2Exp', 'X2Exp', 'X2Exp', 'X2Exp', 'X2Exp', 
         'X1X2Add', 'X1X2Add', 'X1X2Add', 'X1X2Add', 'X1X2Add', 'X1X2Add', 'X1X2Add', 'X1X2Add', 'X1X2Prod', 'X1X2Prod', 'X1X2Prod', 'X1X2Prod', 'X1X2Prod', 'X1X2Prod', 'X1X2Prod', 'X1X2Prod']

        intervals: (int) the number of "same" named predicates to be learned, in a list of predicates
                    we will have ['X1>a', 'X1<b',...], this int provides the total number and helps to isolate 
                    the max weight for those predicates, and then compare the value to 
        '''
        wt = tf.get_collection(self.name)
        #print(wt)
        if len(wt)<2:
            return ''
        
        if '_AND' in wt[0].name:
            w_andt = wt[0]
            w_ort  = wt[1]
        else:
            w_andt = wt[1]
            w_ort  = wt[0]
        #w_and,w_or = session.run( [w_andt,w_ort] )
        #w_and = sharp_sigmoid_np(w_andt,self.sig)
        #w_or = sharp_sigmoid_np(w_ort,self.sig)

        #IDEA: softmax the weights to get likelihoods
        predcount = w_andt.shape[1]
        preddefinitions = w_andt.shape[0]
        pred_max_weights = []
        threshold = tf.constant([0.2])
        for d in range(preddefinitions):
            class_def_weights = []
            for k in range(0,predcount,2*intervals):
                #print(k)
                temp = tf.math.sigmoid( w_andt[d,k:k+2*intervals])
                max_weight = tf.reduce_max(temp)
                max_index = tf.math.argmax(temp)
                less_pred = temp[max_index]
                only_true_preds = tf.math.greater(max_weight, threshold)
                sort_tensor_pred = tf.sort(temp)
                if True: #need a tensor method to make the comparison
                    class_def_weights.append(sort_tensor_pred)
                else:
                    class_def_weights.append(0)
            pred_max_weights.append(class_def_weights)
        return pred_max_weights

    def getTensorIntervalWeights(self, pred_name, cnt_vars ,intervals = 2, threshold =0.2):
        '''
        method gets a list of probabilities for a given set of same named predicates
        ie. ['X1', 'X1', 'X1', 'X1', 'X1', 'X1', 'X1', 'X1'] and their respective weighted probabilities
        of being true
        pred_name: (string) name of the predicate we are looking for 'X1'
        cnt_var: (list) of predicate names, used to get the starting index in the tensor
        interals: the number of named predicates used in learning
        '''
        wt = tf.get_collection(self.name)
        #print(wt)
        if len(wt)<2:
            return ''
        
        if '_AND' in wt[0].name:
            w_andt = wt[0]
            w_ort  = wt[1]
        else:
            w_andt = wt[1]
            w_ort  = wt[0]
        predcount = w_andt.shape[1]
        preddefinitions = w_andt.shape[0]
        pred_max_weights = []
        threshold = tf.constant([0.2])
        start_index_pred_name = cnt_vars.index(pred_name)
        end_index_pred_name = start_index_pred_name + 2*intervals
        
        w_and_prob =w_andt# tf.math.sigmoid(w_andt)#, self.sig)
        w_or_prob = w_ort#tf.math.sigmoid(w_ort)#, self.sig)

        return w_and_prob[start_index_pred_name:end_index_pred_name], w_or_prob[start_index_pred_name:end_index_pred_name]


        

        

    def getSimpleFunc(self, session,names=None,threshold=.2,print_th=True):
        '''
        This function prints out the predicates currently in the body of the
        target predicates we are trying to learn
        session: tensor session which has the current state of the model
        names: name of all continuous predicates predicate
        '''
        
        wt = tf.get_collection(self.name) #list of tensors [tf(2,80), tf(1,2)]
        #print(wt)
        if len(wt)<2:
            return ''
        
        if '_AND' in wt[0].name:
            w_andt = wt[0]
            w_ort  = wt[1]
        else:
            w_andt = wt[1]
            w_ort  = wt[0]
        w_and,w_or = session.run( [w_andt,w_ort] )
        w_and = sharp_sigmoid_np(w_and,self.sig)
        w_or = sharp_sigmoid_np(w_or,self.sig)
        #print(w_and)
        #print(w_or)

        clauses = []
        terms=[]

        germs=[]

        for k in range(w_or[0,:].size):
            if w_or[0,k]>threshold:
                
                terms=[]

                #germs=[]

                for v in range( w_and[k,:].size):
                    if w_and[k,v]>threshold:
                        if names is None:
                            terms.append( 'I_%d'%(v+1))
                        else:
                            terms.append( names[v])
                            germs.append(names[v])


                        if print_th and w_and[k,v]<.95:
                                terms[-1] = '[%.2f]'%(w_and[k,v]) + terms[-1]
                        if print_th:
                            germs[-1] = '[%.2f]'%(w_and[k,v]) + germs[-1]
                #print("-----------------------------------------------------------------------")
                #print(terms)
                #print(germs)
                #print("-----------------------------------------------------------------------")
                s = ','.join(terms)
                #print(s)
                if print_th and w_or[0,k]<.9999:
                    clauses.append( '\t :- [%.2f] ('%(w_or[0,k]) + s +' )')
                else:
                    clauses.append( '\t :- ('%(w_or[0,k]) + s +' )')
        return germs


    def get_item_contribution(self,session,names,threshold=.2 ):
        items = {}

        wt = tf.get_collection(self.name)
        
        if len(wt)<2:
            return ''
        
        if '_AND' in wt[0].name:
            w_andt = wt[0]
            w_ort  = wt[1]
        else:
            w_andt = wt[1]
            w_ort  = wt[0]

        w_and,w_or = session.run( [w_andt,w_ort] )
        w_and = sharp_sigmoid_np(w_and,self.sig)
        w_or = sharp_sigmoid_np(w_or,self.sig)
    

         
        max_or = np.max(w_or[0,:]) + 1e-3
        max_or=1.
        for k in range(w_or[0,:].size):
            if w_or[0,k]>threshold:
                max_and = np.max(w_and[k,:]) + 1e-3
                max_and=1
                for v in range( w_and[k,:].size):
                    if w_and[k,v]>threshold:
                        if names is None:
                            tn = 'I_%d'%(v+1)
                        else:
                            tn=  names[v]

                        # if tn in items:
                        #     items[tn] = max( items[tn], w_and[k,v] )
                        # else:
                        #     items[tn] =   w_and[k,v] 
                        if tn in items:
                            items[tn] = max( items[tn],(w_or[0,k] * w_and[k,v] /max_or)/max_and  )
                        else:
                            items[tn] = (w_or[0,k] * w_and[k,v] /max_or)/max_and

        return items

           