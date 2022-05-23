import  numpy as np
from itertools import product,permutations
from collections import OrderedDict 
from datetime import datetime
from collections import Counter

variable_list=['A','B','C','D','E','F','G','H','I','J','K','L','M','N']

#####################################################
def gen_all_orders2( v, r,var=False):
    '''
    creates a list of ordered permutations of variable sets
    v: set {}, list of different variables:
    r: list []
    '''
    if not var:
        inp = [v[i] for i in r]
    else:
        inp = [v[i[0]] for i in r]
    #Cartesian product of input iterables. Roughly equivalent to nested for-loops in a generator expression. 
    #For example, product(A, B) returns the same as ((x,y) for x in A for y in B).    
    p = product( *inp) #creates an iterator
    return [kk for kk in p]


def subtraction_minmax_helper(list_one,list_two):
    most_max = 0
    most_min = 0
    for x1 in list_one:
        for x2 in list_two:
            temp = x1-x2
            if temp > most_max:
                most_max = temp
            elif temp < most_min:
                most_min = temp
    return most_max, most_min

def product_minmax_helper(list_one, list_two):
    most_min=0
    most_max=0
    for x1 in list_one:
        for x2 in list_two:
            temp = x1*x2
            if temp > most_max:
                most_max = temp
            elif temp < most_min:
                most_min = temp
    return most_max, most_min


#####################################################
#####################################################
#####################################################

class PredCollection:
    '''
    Pred collection object which contains lists and for constant names, predicate names, and continuous values
    '''
    def __init__(self, constants  ):
        self.constants = constants
        self.preds = []
        self.cnts=[]
        self.preds_by_name=dict({})
    def get_constant_list( self,pred , vl ):
        '''
        Create constant list
        pred: class Predicate; 
        vl: list of variables [A,B,C...]
        '''
        Cs=dict( { k:[] for k in self.constants.keys()})
        for i,cl in enumerate( pred.arguments+pred.variables ): #for us both lists are empty, so everything is skipped and Cs is returneds
            Cs[cl[0]].append( vl[i])
            if cl[0]=='N':
                if pred.use_M: 
                    Cs['N'].append('M_' + vl[i] )
                if pred.use_P: 
                    Cs['N'].append('P_' + vl[i] )
                    
            if cl[0]=='L':
                
                if pred.use_tH:
                    Cs['L'].append('H_'+vl[i])
                    Cs['C'].append('t_'+vl[i])
                
                if pred.use_Th:
                    Cs['C'].append('h_'+vl[i])
                    Cs['L'].append('T_'+vl[i])
        
        return Cs
    def get_continuous_var_names(self,p,thDictGT,thDictLT):
        terms=[]
        for v in self.cnts:
            
            cond1 = p.exc_cnt is not None and v.name in p.exc_cnt
            cond2 = p.inc_cnt is not None and v.name not in p.inc_cnt
            if not (cond1 or cond2):
                terms.extend( v.get_terms(thDictGT[v.name],thDictLT[v.name]) )
        return terms
    def get_continuous_var_names_novar(self,p):
        terms=[]
        for v in self.cnts:
            
            cond1 = p.exc_cnt is not None and v.name in p.exc_cnt
            cond2 = p.inc_cnt is not None and v.name not in p.inc_cnt
            if not (cond1 or cond2):
                terms.extend( v.get_terms_novar() )
        return terms

    #function to list out the body of a class predicate
    def get_continuous_predbody_names(self, p):
        for v in self.cnts:
            print(v)

    def add_counter( self):
        self.add_pred(name='CNT',arguments=['N'] , variables=[] )   
    def add_continuous( self, name,no_lt, no_gt,lt_init=None,gt_init=None,dim=1,max_init=1,min_init=0):
        '''
        add continuous predicates to the KB  of the from (X1>a)
        params: see params for COntinousVar
        '''
        self.cnts.append( ContinousVar(name,no_lt,no_gt,lt_init,gt_init,dim,max_init,min_init))
    def add_continuous_nl(self, name,no_lt, no_gt,lt_init=None,gt_init=None,dim=1,max_init=1,min_init=0):
        '''
        add continuous non-linear transformed predicates to the KB  of the from (sinX1>a, squareX1<b, etc.)
        params: see params for COntinousVar

        ISSUE: As sine is a periodic function, we need be sure the value for min/max are correct
                same for any non-linear function
        TEMP: Just commented out some of the predicates for the purpose of the simple function path tracing
        '''
        self.cnts.append( ContinousVar(name+'Square',no_lt,no_gt,lt_init,gt_init,dim,np.power(max_init,2),np.power(min_init,2)))
        self.cnts.append( ContinousVar(name+'Exp', no_lt,no_gt,lt_init,gt_init,dim,np.exp(max_init),np.exp(min_init)))

        min_sin = 0
        max_sin = 0
        #this is hacky fix, a better option is to identify this in ILPRLEngine and pass in the known min, max for sine
        for i in np.arange(float(min_init),float(max_init), 0.2):
            if np.sin(i) < min_sin:
                min_sin = np.sin(i)
            if np.sin(i) > max_sin:
                max_sin = np.sin(i)
        
        #self.cnts.append( ContinousVar(name+'Sine', no_lt, no_gt, lt_init, gt_init, dim, max_sin, min_sin))
        #
    def add_continuous_exp(self, name,no_lt, no_gt,lt_init=None,gt_init=None,dim=1,max_init=1,min_init=0):
        self.cnts.append( ContinousVar(name+'Exp', no_lt,no_gt,lt_init,gt_init,dim,np.exp(max_init),np.exp(min_init)))
    
    def add_continuous_square(self, name,no_lt, no_gt,lt_init=None,gt_init=None,dim=1,max_init=1,min_init=0):
        self.cnts.append( ContinousVar(name+'Square',no_lt,no_gt,lt_init,gt_init,dim,np.power(max_init,2),np.power(min_init,2)))

    def add_continuous_sin(self, name, no_lt, no_gt, lt_init=None, gt_init=None, dim=1, max_init=1, min_init=1):
        min_sin = 0
        max_sin = 0
        #this is hacky fix, a better option is to identify this in ILPRLEngine and pass in the known min, max for sine
        for i in np.arange(float(min_init),float(max_init), 0.2):
            if np.sin(i) < min_sin:
                min_sin = np.sin(i)
            if np.sin(i) > max_sin:
                max_sin = np.sin(i)
        
        self.cnts.append( ContinousVar(name+'Sine', no_lt, no_gt, lt_init, gt_init, dim, max_sin, min_sin))


    def add_continuous_gi(self, name, no_lt, no_gt, lt_init=None, gt_init=None, dim=1, max_init=1, min_init=0):
        g_neg = -1 * 9.8
        Iplus1 = 0.001+1
        self.cnts.append( ContinousVar(name+'multGravity',no_lt,no_gt,lt_init,gt_init,dim,(max_init *g_neg),(min_init*g_neg)))
        self.cnts.append( ContinousVar(name+'divInertia',no_lt,no_gt,lt_init,gt_init,dim,(max_init/Iplus1),(min_init/Iplus1)))


    def add_continuous_operations(self, name1, name2, no_lt, no_gt, lt_init = None, gt_init = None, dim= 1, max_init1=1, min_init1=0, max_init2=1, min_init2 = 0):
        '''
        Add continous predicats which indicate operations between different variables
        at the moment this limited to just two variable pairings
        params:  see params for Continuous featues, note we pass in 2 feature names and 2 min/max values as we
                are looking at operations between 2 features
        ISSUE: subtraction is not commutative, this needs to be considered
        ISSUE: The current implementation is flawed in the sense its looking for operations on
                the variables X1..Xn prior to transforming the feature, say sine(X1).
                 This leads to operations being missed when say one transformation x1^2 is significantly larger than sinX, or logX.
                 We nee to consider that some operations will be weighted more than other...
                Is there a way to either deal with this in the creating the KB (see class background)
                so that the operation predicates include all sets of transformations, may be too exuastive
                A better approach would be to include another DNF after we've learned the transofmrations [expX1, sinX2]
                which looks at operation predicates only...non-trivial
        '''

        self.cnts.append(ContinousVar(name1 + name2 + 'Add', no_lt, no_gt, lt_init, gt_init, dim, max_init1 + max_init2, min_init1 + min_init2))
        #self.cnts.append(ContinousVar(name1 + name2 + 'Sub', no_lt, no_gt, lt_init, gt_init, dim, max_init1 + max_init2, min_init1 - min_init2))
        #self.cnts.append(ContinousVar(name1 + name2 + 'Prod', no_lt, no_gt, lt_init, gt_init, dim, max_init1 * max_init2, min_init1 * min_init2))

    def add_continuous_add(self, name1, name2, no_lt, no_gt, lt_init = None, gt_init = None, dim= 1, max_init1=1, min_init1=0, max_init2=1, min_init2 = 0):
        self.cnts.append(ContinousVar(name1 + name2 + 'Add', no_lt, no_gt, lt_init, gt_init, dim, max_init1 + max_init2, min_init1 + min_init2))




    def add_continuous_sub(self, name1, name2, no_lt, no_gt, lt_init = None, gt_init = None, dim= 1, max_init1=1, min_init1=0, max_init2=1, min_init2 = 0, both = True, first=True):
        list_one = [max_init1,min_init1]
        list_two = [max_init2,min_init2]
        
        if both:
            mmax1, mmin1 = subtraction_minmax_helper(list_one, list_two)
            self.cnts.append(ContinousVar(name1 + name2 + 'Sub', no_lt, no_gt, lt_init, gt_init, dim, mmax1, mmin1))
            mmax2, mmin2 = subtraction_minmax_helper(list_two, list_one)
            self.cnts.append(ContinousVar(name2 + name1 + 'Sub', no_lt, no_gt, lt_init, gt_init, dim, mmax2, mmin2))
        else:
            if first:
                mmax, mmin = subtraction_minmax_helper(list_one, list_two)

                self.cnts.append(ContinousVar(name1 + name2 + 'Sub', no_lt, no_gt, lt_init, gt_init, dim, mmax, mmin))
            else:
                mmax, mmin = subtraction_minmax_helper(list_two, list_one)
                self.cnts.append(ContinousVar(name2 + name1 + 'Sub', no_lt, no_gt, lt_init, gt_init, dim, mmax, mmin))



    def add_continuous_prod(self, name1, name2, no_lt, no_gt, lt_init = None, gt_init = None, dim= 1, max_init1=1, min_init1=0, max_init2=1, min_init2 = 0):
        list_one = [max_init1,min_init1]
        list_two = [max_init2,min_init2]
        prod_max, prod_min = product_minmax_helper(list_one, list_two)
        self.cnts.append(ContinousVar(name1 + name2 + 'Prod', no_lt, no_gt, lt_init, gt_init, dim,prod_max, prod_min))# max_init1 * max_init2, min_init1 * min_init2))
    
    

    def add_number_preds(self,ops=['incN','zeroN', 'lteN','eqN', 'gtN'   ]):
        
        if 'incN' in ops:
            self.add_pred(name='incN'      ,arguments=['N','N'] , variables=[] )   

        if 'addN' in ops:
            self.add_pred(name='addN'      ,arguments=['N','N','N'] , variables=[] )   

        if 'zeroN' in ops:
            self.add_pred(name='zeroN'      ,arguments=['N'] , variables=[] )   
        if 'lteN' in ops:
            self.add_pred(name='lteN'      ,arguments=['N','N'] , variables=[] )   

        if 'gtN' in ops:
            self.add_pred(name='gtN'      ,arguments=['N','N'] , variables=[] )   

        if 'eqN' in ops:
            self.add_pred(name='eqN'      ,arguments=['N','N'] , variables=[] )   
    def add_list_preds(self , ops=['emptyL','eqC', 'eqL','singleL'  ] ):
        
        if 'LI' in ops:
            self.add_pred(name='LI'      ,arguments=['L', 'C','N'] , variables=[] )  # [_|t]-> ?
        if 'LL' in ops:
            self.add_pred(name='LL'      ,arguments=['L', 'N'] , variables=[] )  # [_|t]-> ?
        
        if 'eqC' in ops:
            self.add_pred(name='eqC'      ,arguments=['C','C'] , variables=[] )  # [_|t]-> ?
        if 'eqLC' in ops:
            self.add_pred(name='eqLC'      ,arguments=['L','C'] , variables=[] )  # [_|t]-> ?
        if 'emptyL' in ops:
            self.add_pred(name='emptyL'      ,arguments=['L'] , variables=[] )  # [_|t]-> ?
        if 'eqL' in ops:
            self.add_pred(name='eqL'     ,arguments=['L','L'] , variables=[] )   
        if 'singleL' in ops:
            self.add_pred(name='singleL'     ,arguments=['L'] , variables=[] )   
        if 'appendL' in ops:
            self.add_pred(name='appendL'     ,arguments=['L','L','L'] , variables=[] )   
        if 'appendC1' in ops:
            self.add_pred(name='appendC1'     ,arguments=['L','C','L'] , variables=[] )   
        if 'appendC2' in ops:
            self.add_pred(name='appendC2'     ,arguments=['C','L','L'] , variables=[] )   
    def add_pred( self,**args):
        '''
        the method is responsible for creating our class predicates
        params: the arg are for the class Predicate
        '''
        p = Predicate( **args)
        self.preds.append(p)
        self.preds_by_name[p.name] = p
        return p
    def __len__(self):
        return len(self.preds)
    def __getitem__(self, key):
        if type(key) in (str,):
            return self.preds_by_name[key]
        else:
            return self.preds[key]
    def apply_func_args(self,Cs):
        
        if 'C' in Cs:
            for i,v in enumerate(Cs['C']):
                if v.startswith('t_'):
                    if v=='t_':
                        Cs['C'][i] = ''
                    else:
                        Cs['C'][i]=v[-1]
        if 'L' in Cs:
            for i,v in enumerate(Cs['L']):
                if v.startswith('H_'):
                    if v=='H_':
                        Cs['L'][i] = ''
                    else:
                        Cs['L'][i]= v[2:-1]
        if 'C' in Cs:
            for i,v in enumerate(Cs['C']):
                if v.startswith('h_'):
                    if v=='h_':
                        Cs['C'][i] = ''
                    else:
                        Cs['C'][i]=v[2]
        if 'L' in Cs:
            for i,v in enumerate(Cs['L']):
                if v.startswith('T_'):
                    if v=='T_':
                        Cs['L'][i] = ''
                    else:
                        Cs['L'][i]= v[3:]
        if 'N' in Cs:
            for i,v in enumerate(Cs['N']):
                if type(v) in (str,) and v.startswith('P_'):
                    Cs['N'][i] = '%d'%( int(v[2:])+1)
                if type(v) in (str,) and v.startswith('M_'):
                    val= max(0,int(v[2:])-1)
                    Cs['N'][i] = '%d'%(val)
                    
        return Cs

    def initialize_predicates(self):
        '''
        
        '''
        
        t1=  datetime.now()

        def map_fn( pair_arg,pair_val ,pred ):
             
             
            in_indices=[]
            L = 0
            Cs = self.get_constant_list(pred,pair_arg+pair_val)
            Cs = self.apply_func_args(Cs)

            for p in self.preds:
                
                
                # exclude some predciates
                
                if pred.inc_preds is not None and p.name not in pred.inc_preds:
                    L +=  p.pairs_len 
                    continue
                if pred.exc_preds is not None and p.name in pred.exc_preds:
                    L +=  p.pairs_len 
                    continue
                
                name_set =  gen_all_orders2( Cs , p.arguments,var=True)
                for i,n in  enumerate(name_set):
                    if i in pred.exc_term_inds[p.name]:
                        continue
                    try:
                        # ind = p.pairs.index(n)
                        ind = p.rev_pairs_index[n]
                        in_indices.append(ind+L)
                    except:
                        in_indices.append(-1)
                L +=  p.pairs_len 
                
            return in_indices
       
        # fill pred.pairs , pred.inp_list , pred.Lx , pred.Lx_details, pred.self_termIndex
        for pred in self.preds:  #(class_1, class_2, class_3)
            
            #if we have inputs for our predicate, say pred(A,B,C) this statement
            # would create different pairs of input posistions, pred(A,C,B), pred(B,C,A) etc. 
            if pred.pairs is None:
                pred.pairs = gen_all_orders2(self.constants,pred.arguments) #[()] its empty as there are no arguments
                pred.pairs_len = len( pred.pairs ) # pair_lens = 1
                # pred.pairs.sort()  
            #creates a ordered dictionary index for the input var pairings 1 = (A,A), 2 = (A,B)
            pred.rev_pairs_index=OrderedDict()
            for ii in range(len( pred.pairs)):
                pred.rev_pairs_index[ pred.pairs[ii]] = ii
            #
            pred.Lx_details =[]
            Cs = self.get_constant_list(pred,variable_list) #class_1, [A,B,C,...]
            
            for p in self.preds: #(class_1, class_2, class_3)
                pred.exc_term_inds[p.name]=[]    
                if pred.inc_preds is not None and p.name not in pred.inc_preds:
                    pred.Lx_details.append(0) #will add 3 0s, [0,0,0]
                    continue
                if pred.exc_preds is not None and p.name in pred.exc_preds:
                    pred.Lx_details.append(0)
                    continue

                name_set =  gen_all_orders2( Cs , p.arguments,var=True)
                Li=0
                
                for i,n in enumerate(name_set):
                    term = p.name + '(' + ','.join(n)+')'
                    pcond = False
                    for c in pred.exc_conds:
                        if p.name == c[0] or c[0]=='*':
                            cl=Counter(n)
                            l = list(cl.values())
                            if c[1]=='rep1':
                                if max(l)>1:
                                    pcond=True
                                    break
                            if c[1]=='rep2':
                                if max(l)>2:
                                    pcond=True
                                    break
                    if term not in pred.exc_terms and not pcond:
                        Li+=1
                        pred.inp_list.append( term)
                    else:
                        pred.exc_term_inds[p.name].append(i)
                     
                pred.Lx_details.append(Li)
            
            if pred.use_neg:
                negs=[]
                for k in pred.inp_list:
                    negs.append( 'not '+k)  
                pred.inp_list.extend(negs)
            pred.Lx = sum(pred.Lx_details) # 0 = 0+0+0
        
        

        
          
        self.values_sizes = OrderedDict({})
        for pred in self.preds:
            self.values_sizes[pred] = pred.pairs_len
        self.InputIndices=dict({})
         
        
        for p in self.preds:
            
            pairs_var = gen_all_orders2( self.constants , p.variables)
            len_pairs_var = len(pairs_var)

            self.InputIndices[p.name] = np.zeros( [ p.pairs_len, len_pairs_var , p.Lx], np.int64)
            
            if True:
                print('******************************************************************')
                print('predicate [%s] parameters :'%(p.name) )
                print('Lx :',p.Lx)
                print('Lx Details',p.Lx_details )
                print('input index shape : ', self.InputIndices[p.name].shape)
                print('******************************************************************')

            if p.pFunc is not None:    
                for i in range( p.pairs_len ):
                    for j in range( len_pairs_var ):
                        inds = map_fn( p.pairs[i],pairs_var[j], p )
                        self.InputIndices[p.name][i,j]=inds
                
                self.InputIndices[p.name]  = np.array(self.InputIndices[p.name],np.int64)
            

        
        t2=  datetime.now()
        print ('building background knowledge finished. elapsed:' ,str(t2-t1))
    def get_terms_indexs(self,pred_name,terms):
        inds=[]
        terms=terms.split(', ')
        for t in terms:
            inds.append(self.preds_by_name[pred_name].inp_list.index(t))
        return np.array(inds,dtype=int)
#####################################################
#####################################################
#####################################################

class PredFunc:
    def __init__(self,name='',trainable=True):
        self.trainable = trainable
        self.name = name
    
    def pred_func(self,xi,xcs=None,t=0):
        pass
    
    def get_func(self,session,names=None,threshold=.1,print_th=True):
        pass
    def getSimpleFunc(self, session,names=None,threshold=.1,print_th=True):
        pass
    def get_item_contribution(self,session,names=None,threshold=.1):
        pass
    def conv_weight_np(self,w):
        return w
    def conv_weight(self,w):
        return w


#####################################################
#####################################################
#####################################################

class ContinousVar:
    '''
    A continuous varibale class object used for all atoms 
    '''
    def __init__(self,name,no_lt,no_gt,lt_init,gt_init,dim=1,max_init=1,min_init=0):
        '''
        name: name of continuous variable (String)
        no_lt: number of less than atoms
        no_gt: number of greater than atoms
        dim: 
        gt_init: array of the greater than initial values for our continuous predicae
        lt_init: array of the less than initial values for our continuous predicae
        max_init: max value of the continous variable 
        min_init: min value of the continuous variable
        '''
        self.name=name
        self.no_lt = no_lt 
        self.no_gt = no_gt 
        self.dim=dim
        self.gt_init = gt_init
        self.lt_init = lt_init
        self.max_init = max_init
        self.min_init = min_init
        #difference for calc of our interval values for lessthan/greatthan predicates
        diff = max_init - min_init

        
        if self.gt_init is None:
            self.gt_init = np.linspace( self.min_init + diff/(1+no_gt) , self.max_init - diff/(1+no_gt) , no_gt  ) #with max(3),min(1), we get [1.4,1.8,2.2,2.6]
        if self.lt_init is None:
            self.lt_init = np.linspace( self.min_init + diff/(1+no_gt) , self.max_init - diff/(1+no_gt) , no_gt  ) #same for gt_init
    
    def get_terms(self,v_gt,v_lt):
        
        terms = []
        for i in range(self.no_gt):
            terms.append( self.name+">%.2f"%v_gt[i] )
        for i in range(self.no_lt):
            terms.append( self.name+"<%.2f"%v_lt[i])
        return terms
    
    def get_terms_novar(self):
        
        terms = []
        for i in range(self.no_gt):
            terms.append( self.name )
        for i in range(self.no_lt):
            terms.append( self.name )
        return terms

###################################################
#####################################################
#####################################################
class Background:

    def __init__(self,predColl ):
        
        self.predColl = predColl
        self.backgrounds = OrderedDict({})
        self.backgrounds_value = OrderedDict({})
        self.examples = OrderedDict({})
        self.examples_value = OrderedDict({})
        self.backgrounds_ind = OrderedDict({})
        self.examples_ind = OrderedDict({})
        self.continuous_vals = OrderedDict({})

        for p in predColl.preds:
            self.backgrounds[p.name] = []
            self.backgrounds_value[p.name] = []
            self.backgrounds_ind[p.name] = []
            self.examples[p.name] = []
            self.examples_value[p.name] = []
            self.examples_ind[p.name] = []

    def add_backgroud( self,pred_name , pair ,value=1): 
        
        if pair not in self.backgrounds[pred_name]:
            self.backgrounds[pred_name].append( pair)
            self.backgrounds_value[pred_name].append(value)
            self.backgrounds_ind[pred_name].append( self.predColl[pred_name].pairs.index(pair))

    def add_example(self,pred_name,pair,value=1):
        if pair not in self.examples[pred_name]:
            self.examples[pred_name].append(pair)
            self.examples_value[pred_name].append(value)
            self.examples_ind[pred_name].append( self.predColl[pred_name].pairs.index(pair))

    def add_all_neg_example(self,pred_name):
        pairs = gen_all_orders2(self.predColl.constants, self.predColl[pred_name].arguments )
        for pa in pairs:
            if  pa not in self.examples[pred_name] :#and pa not in self.backgrounds[pred_name] :
                self.add_example(pred_name,pa,0.)
    
    def add_all_neg_example_ratio(self,pred_name, ratio):
        # pairs = gen_all_orders2(self.predColl.constants, self.predColl[pred_name].arguments )
        pos = np.sum(self.examples_value[pred_name]) 
        max_neg = pos * ratio
        inds= np.random.permutation(self.predColl.pairs_len[pred_name] )
        
        cnt = 0 
        for i in inds:
            if  self.predColl.pairs[pred_name][i]  not in self.examples[pred_name] :#and pa not in self.backgrounds[pred_name] :
                cnt+=1
                self.add_example(pred_name,self.predColl.pairs[pred_name][i],0.)
            if cnt>max_neg:
                break
   
   
   
    def add_number_bg( self,N ,ops=['incN','zeroN', 'lteN','eqN', 'gtN'  ]):
        if 'zeroN' in ops :
            self.add_backgroud( 'zeroN' , ('0',) )
        
        for a in N:
            if 'incN' in ops:
                if  ( str(int(a)+1)) in N:
                    self.add_backgroud('incN', ( a, str(int(a)+1) ) )
            
        for a in N:
            for b in N:
                    if 'addN' in ops:
                        c = str( int(a)+int(b) )
                        if c in N:
                            self.add_backgroud('addN', (a,b,c))
                    
                    if 'eqN' in ops and a==b:
                        self.add_backgroud('eqN', (a , b))
                    if 'lteN' in ops and a<=b:
                        self.add_backgroud('lteN', (a , b))
                    if 'gtN' in ops and a>b:
                        self.add_backgroud('gtN',(a , b))

    def add_list_bg(self, C,Ls , ops=['emptyL','eqC', 'eqL','singleL'  ]):
        if 'LI' in ops:
            for a in Ls:
                for i in range(len(a)):
                    self.add_backgroud( 'LI', (a,a[i], str(i) ) )
        
        if 'LL' in ops:
            for a in Ls:
                self.add_backgroud( 'LL', (a, str(len(a)) ) )
              
        if 'emptyL' in ops:
            self.add_backgroud ('emptyL' , ('',) )
        
        for a in C:
            if 'eqC' in ops:
                self.add_backgroud ('eqC', (a,a) ) 
            if 'eqLC' in ops:
                if a in C and a in Ls:
                    self.add_backgroud ('eqLC', (a,a) ) 

        for a in Ls:
            if 'eqL' in ops:
                self.add_backgroud ('eqL', (a,a) ) 
            
            if len(a) ==1:
                if 'singleL' in ops:
                    self.add_backgroud ('singleL', (a,) ) 

            for b in Ls:
                if a+b in Ls:
                    if 'appendL' in ops:
                        self.add_backgroud ('appendL', (a,b,a+b) ) 
                    if 'appendC1' in ops and len(b)==1:
                        self.add_backgroud ('appendC1', (a,b,a+b) ) 
                    if 'appendC2' in ops and len(a)==1:
                        self.add_backgroud ('appendC2', (a,b,a+b) ) 

        return

    def add_continuous_valuea( self,vdic):
        self.continuous_vals.update(vdic)
    
    def add_continuous_value( self,key,value):
        self.continuous_vals[key]=value

    def add_continuous_value_nl( self,key,value):
        self.continuous_vals[key+'Square']=np.power(value,2)
        #self.continuous_vals[key+'Sine']=np.sin(value)
        self.continuous_vals[key+'Exp']=np.exp(value)
    
    def add_continuous_value_square( self,key,value):
        self.continuous_vals[key+'Square']=np.power(value,2)
    
    def add_continuous_value_sin( self,key,value):
        self.continuous_vals[key+'Sine']=np.sin(value)

    def add_continuous_value_exp( self,key,value):
        self.continuous_vals[key+'Exp']=np.exp(value)

    def add_continuous_value_gi(self, key, value):
        g_neg = -1 * 9.8
        Iplus1 = 0.001+1
        print(value)
        #print(value.type)
        #print(g_neg.type)
        self.continuous_vals[key+'Exp']=np.exp(value)
        self.continuous_vals[key+'multGravity'] = np.array(value) * g_neg
        self.continuous_vals[key+'divInertia'] = np.array(value) / Iplus1

    
    def add_continuous_value_op(self, key1, key2, value1, value2):
        self.continuous_vals[key1 + key2 + 'Add'] = np.add(value1 , value2)
        #self.continuous_vals[key1 + key2 + 'Sub'] = np.subtract(value1 , value2)
        #self.continuous_vals[key1 + key2 + 'Prod'] = np.multiply(value1 , value2)
        '''
        commenting out some predicates for the purpose of tracing the simple function path through the function
        '''
        #print(self.continuous_vals[key1+key2 + 'Add'])

    def add_continuous_value_add(self, key1, key2, value1, value2):
        self.continuous_vals[key1 + key2 + 'Add'] = np.add(value1 , value2)

    def add_continuous_value_prod(self, key1, key2, value1, value2):
        self.continuous_vals[key1 + key2 + 'Prod'] = np.multiply(value1 , value2)
    
    def add_continuous_value_sub(self, key1, key2, value1, value2, both = True, first = True):
        #self.continuous_vals[key1 + key2 + 'Sub'] = np.subtract(value1 , value2)
        #self.continuous_vals[key2 + key1 + 'Sub'] = np.subtract(value2 , value1) 
        if both:
            self.continuous_vals[key1 + key2 + 'Sub'] = np.subtract(value1 , value2)
            self.continuous_vals[key2 + key1 + 'Sub'] = np.subtract(value2 , value1)
        else:
            if first:
                self.continuous_vals[key1 + key2 + 'Sub'] = np.subtract(value1 , value2)
            else:
                self.continuous_vals[key2 + key1 + 'Sub'] = np.subtract(value2 , value1) 


        
    def get_X0(self,pred_name):
        x = np.zeros( [self.predColl[pred_name].pairs_len ] , np.float32 ) 
        x[self.backgrounds_ind[pred_name]]=self.backgrounds_value[pred_name]
        return x
    
    def get_target_data(self,pred_name):
        x = np.zeros( [self.predColl[pred_name].pairs_len ] , np.float32 )
        x[self.examples_ind[pred_name]]=self.examples_value[pred_name]
        return x
        
    def get_target_mask(self,pred_name):
        x = np.zeros( [self.predColl[pred_name].pairs_len ] , np.float32 )
        x[self.examples_ind[pred_name]]=1
        return x

###################################################
#####################################################
#####################################################


class Predicate:
    '''
    Predicate class used to define 
    '''
    
    def __init__(self,name, arguments,variables=[],pFunc=None, inc_preds=None,exc_preds=None,use_cnt_vars=False, use_neg=False,arg_funcs=['tH'],inc_cnt=None,exc_cnt=None,Fam='eq', exc_terms=[],exc_conds=[]):
        '''
        name: name of predicate; [ class_1 ]
        exe_term: empty set {} 
        arity: the number of inputs into our predicate; 0 for class_1
        var_count: number of variables; 0 for class_1...which seems odd
        arguments: []
        variables: these would be input variables class(A,B); [] empty list for us 
        pFunc: class layer; class DNF 
        use_neg:
        exc_cnt:
        inc_cnt:
        inc_preds: ; [] empty list
        exc_preds:
        use_cnt_vars: boolean for ... ; True for us
        exc_conds:
        exc_terms:
        arg_funcs: ; ['tH']
        use_tH:
        use_Th: ; False
        use_M:  ; False
        use_P:  ; False
        Fam: string : 'eq'
        rev_pairs_index:
        pairs:
        pairs_len:
        inp_list:
        Lx:
        Lx_details: 
        '''
        self.name=name 
        self.exc_term_inds={}
        self.arity=len(arguments)
        self.var_count = len(variables)
        self.arguments = arguments 
        self.variables = variables

        self.pFunc=pFunc
        self.use_neg = use_neg
        self.exc_cnt=exc_cnt
        self.inc_cnt=inc_cnt
        
        self.inc_preds=inc_preds
        self.exc_preds=exc_preds
        self.use_cnt_vars = use_cnt_vars

        self.exc_conds=exc_conds
        self.exc_terms=exc_terms
        
        self.arg_funcs = arg_funcs
        self.use_tH = ('tH'in arg_funcs)
        self.use_Th = ('Th'in arg_funcs)
        self.use_M = ('M'in arg_funcs)
        self.use_P = ('P'in arg_funcs)
        
        self.Fam = Fam
        self.rev_pairs_index = None
        self.pairs = None
        self.pairs_len = None
        self.inp_list=[]
        self.Lx=0
        self.Lx_details=[]
        
    def get_term_index(self,term ):

        if not 'not ' in term:
            ind = self.inp_list.index(term)
        else:
            ind = self.inp_list.index(term[4:]) + self.Lx

        return ind