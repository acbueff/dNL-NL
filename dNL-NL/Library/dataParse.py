from operator import sub
import numpy as np
import pandas as pd
import re
from Library.mylibw import read_by_tokens
from Library.DNF import DNF
from Library.CNF import CNF
from Library.PredicateLibV5 import PredFunc
from Library.PredicateLibV5 import *
from numpy import int64
from Library.NLILPRLEngine import *
import argparse
from sklearn.metrics import accuracy_score


def discreteBetterClassifier(y, A, bin, size):
    ''''
    Performs equalwidth binning on the target output for a dataset
    params:
    y: target output vector
    A: data values matrix
    bin: number of discrete classes on our target output
    size: number of instances in the dataset
    '''
  #y = func(A)
  #print("dawg")
  #print(y.shape)
    yout = np.zeros([size,1])
    amax = np.amax(y)
    amin = np.amin(y)
    diff = amax - amin
    rangevalues = []
    rangevalues.append(amin)
    diffbin = diff/bin
    for i in range(bin):
        binvalue = ((i+1)*diffbin + amin)
        rangevalues.append(binvalue)
    print(rangevalues)


    for row, yrow in list(enumerate(y)):
    #print("SORTING")
    #print(yrow)
    #print(row)
        newy = 0
        discreteY = 0 
        for  j in range(bin):
            jplus1 = j+1
            if jplus1 == bin:
                if rangevalues[j] <= yrow:# and yrow <= rangevalues[jplus1]:
                    newy = discreteY
            #     print(newy)
                    break
            else:
                if rangevalues[j] <= yrow and yrow < rangevalues[jplus1]:
                    newy = discreteY
            #    print(newy)
                    break
            discreteY+=1
    #print("SORTED")
        yout[row,0] = newy
    #print("Made it here")
    #print(yrow)
    #print(row)
    #print(y.shape)

    return yout, A

def discreteEqualFreqClassifier(y, A, bin, size):
    yout = np.zeros([size,1])
    amax = np.amax(y)
    amin = np.amin(y)
    df =pd.DataFrame(y, columns=['target'])
    local_labels = []
    for i in range(bin):
        local_labels.append(i)

    qc, bins= pd.qcut(df['target'], q=bin, retbins=True, precision=0, labels=local_labels)
    print(df)
    print(qc)
    print(bins)
    yout = qc[0]
    print(yout)
    numpy_yout = qc.to_numpy()
    best_yout = np.reshape(numpy_yout,(-1,1))
    print("yup")
    return best_yout, A

def parseTransformation(session, predColl, args, pred_parse_list, var_list):
    '''
    select best predicates with large enough weights from NN
    params:
    session: tf session
    predColl: datastructure for all preds/values/etc
    args: NLILRLEngine args
    pred_parse_list: a dict for counting the number of preds over a certain weight
    var_list: list of variable names
    '''
    flag_count = True
    parse_metric = 0.95
    var_operation_count = len(var_list) - 1
    counter = 0
    operation_per_var_pair_list = []
    while(flag_count):
        for p in predColl.preds:
            #print(p.name)
            #print(p.inp_list)
            cnt_vars = predColl.get_continuous_var_names_novar(p)
            if p.pFunc is not None:
                #def get_func(self,session,names,threshold=.2,print_th=True):
                s = p.pFunc.getSimpleFunc( session,cnt_vars,threshold=args.W_DISP_TH)
                #print(s)
                #print("end")
                numbers =[]
                preds = []
                for atom in s:
                    #print(atom)
                    for word in re.split(r'\[|\]',atom):
                        if word.startswith('1.') or word.startswith('0.'):
                            numbers.append(float(word))
                        elif len(word) > 1:
                            preds.append(word)
                #print(numbers)
                #print(preds)
                for i in range(len(numbers)):
                    if numbers[i] > parse_metric:
                        pred_parse_list[preds[i]] += 1
                        #flag_count = False
                #incase no values measured up
        #if no preds were counted then the weight metric requirment is too high and needs to be smaller 
        if flag_count:
            for keys in pred_parse_list.keys():
                if pred_parse_list[keys] > 0:
                    if var_operation_count <= 1: #this means either single or double variables
                        flag_count=False
                        break
                    elif counter == var_operation_count:
                        flag_count=False
                        break
                    else:
                        var_string = keys[0:4]
                        firstvar = var_string[0:2]
                        secondvar= var_string[2:]
                        rev_var_string = secondvar + firstvar #get all operations between these two variables
                        if var_string not in operation_per_var_pair_list and rev_var_string not in operation_per_var_pair_list:
                            operation_per_var_pair_list.append(var_string)
                            operation_per_var_pair_list.append(rev_var_string)
                            counter += 1
            parse_metric = parse_metric - 0.05
            if parse_metric < 0:
                flag_count=False

    #if flag_count:      

    #print(pred_parse_list)
    max_pred_list =[]
    for v in var_list:
        max_pred =''
        max_pred_value =0
        for keys in pred_parse_list.keys():
            if v in keys:
                if pred_parse_list[keys] > max_pred_value:
                    max_pred_value = pred_parse_list[keys]
                    max_pred = keys
                elif pred_parse_list[keys] == max_pred_value:
                    if max_pred == v:
                        max_pred = keys
                    
        max_pred_list.append(max_pred)
    #print(max_pred_list)
    if len(max_pred_list) == 0: #hack fix, need to ensure at least something is extracted
        max_pred_list.append('X1X2Add')
    max_pred_list = list(dict.fromkeys(max_pred_list))
    return max_pred_list    



def parseTransformationOLD(session, predColl, args, pred_parse_list, var_list):
    '''
    select best predicates with large enough weights from NN
    params:
    session: tf session
    predColl: datastructure for all preds/values/etc
    args: NLILRLEngine args
    pred_parse_list: a dict for counting the number of preds over a certain weight
    var_list: list of variable names
    '''
    flag_count = True
    parse_metric = 0.95
    while(flag_count):
        for p in predColl.preds:
            #print(p.name)
            #print(p.inp_list)
            cnt_vars = predColl.get_continuous_var_names_novar(p)
            if p.pFunc is not None:
                #def get_func(self,session,names,threshold=.2,print_th=True):
                s = p.pFunc.getSimpleFunc( session,cnt_vars,threshold=args.W_DISP_TH)
                #print(s)
                #print("end")
                numbers =[]
                preds = []
                for atom in s:
                    #print(atom)
                    for word in re.split(r'\[|\]',atom):
                        if word.startswith('1.') or word.startswith('0.'):
                            numbers.append(float(word))
                        elif len(word) > 1:
                            preds.append(word)
                #print(numbers)
                #print(preds)
                for i in range(len(numbers)):
                    if numbers[i] > parse_metric:
                        pred_parse_list[preds[i]] += 1
                        flag_count = False
                #incase no values measured up
        #if no preds were counted then the weight metric requirment is too high and needs to be smaller
        if flag_count:
            parse_metric = parse_metric - 0.05
            if parse_metric < 0:
                flag_count=False
    #if flag_count:      

    #print(pred_parse_list)
    max_pred_list =[]
    for v in var_list:
        max_pred =''
        max_pred_value =0
        for keys in pred_parse_list.keys():
            if v in keys:
                if pred_parse_list[keys] > max_pred_value:
                    max_pred_value = pred_parse_list[keys]
                    max_pred = keys
                elif pred_parse_list[keys] == max_pred_value:
                    if max_pred == v:
                        max_pred = keys
                    
        max_pred_list.append(max_pred)
    #print(max_pred_list)
    if len(max_pred_list) == 0: #hack fix, need to ensure at least something is extracted
        max_pred_list.append('X1X2Add')
    return max_pred_list

def transformDataset(target_y, data, pred_parse_list, var_list):
    '''
    transform dataset based on the learned predicates from the transformation dNL unit
    params:
    target_y: real valed target vector
    data: origional matrix of input values
    pred_parse_list: list of predicates for transforming the origional data values
    var_list: list of variable names
    '''
    var_index_list ={}
    for i, v in enumerate(var_list):
        var_index_list[v] = i
    
    for k in pred_parse_list:
        for v in var_list:
            if k == v:
                continue
            elif k.startswith(v):
                index = var_index_list[v]
                atom = re.split(v,k)
                if atom[1] == 'Exp':
                    print('exp that data')
                    data[:,index] = np.exp(data[:,index])
                elif atom[1] == 'Sine':
                    print('sine that data')
                    data[:,index] = np.sin(data[:,index])
                elif atom[1] == 'Square':
                    print('square that data')
                    data[:,index] = np.power(data[:,index],2)
    return target_y, data

def operationDataset(target_y, data, pred_parse_list, var_list):
    var_index_list ={}
    for i, v in enumerate(var_list):
        var_index_list[v] = i
    
    for k in pred_parse_list:
        print('derp')
            

def printFunction(varlist, operationlist, transformationlist):
    '''
    simply print the non-linear function
    '''
    func_string =''
    for i, v in enumerate(varlist):
        for t in transformationlist:
            if t.startswith(v):
                func_string += " " + t
                break
        if i == len(varlist)-1:
            break
        for o in operationlist:
            func_string += " " + o
    print(func_string)
    return func_string
  
def orderlistByOperations(operationlist):
  '''
  takes a list of operations and orders them
  '''
  new_list = []
  newnew_set =set()
  for k in operationlist:
    #print(k)
    if k.endswith(('Prod', 'Div')):
      #print("true "  + k)
      new_list.append(k)
      #print(new_list)
      newnew_set = set(new_list)
  for j in operationlist:
    if j.endswith(('Add','Sub')):
      #print("true low"  + j)
      new_list.append(j)
      #print(new_list)
      newnew_set = set(new_list)
      #print(newnew_set)
  return newnew_set, new_list 


def operationOrder(y, data, pred_ops_list, tran_list, var_list):
  '''
  this function calculates the operations on the data and compares the output
  to the origional target y and returns the loss
  params
  y: target vector of real valus
  data: matrix of input values
  pred_ops_list: list of operation preds used to calc the nl function
  tran_list: list of transformations, UNUSED
  var_list: list of variable names
  '''
  var_index_list ={}
  for i, v in enumerate(var_list):
      var_index_list[v] = i
  ops_list = []
  data_out = []
  pred_ops_order , _ = orderlistByOperations(pred_ops_list)
  #print(pred_ops_order)
  for k in pred_ops_order:
      print("lets go "+k)
      for v1 in var_list: #['X1','X2']
        if k == v1:
          print('do nothing')
          continue
        elif k.startswith(v1): #'X2X1Sub'.startswith('X1)
          index1 = var_index_list[v1]
          atom1 = re.split(v1,k) #'X1Sub'
          if len(data_out) == 0:
            data_out = data[:,index1]
          #print(atom1)         
          #print(atom1[1])
          #print(index1)
          for v2 in var_list:
            if atom1[1].startswith(v2):#'X1Sub'.startswith('X1')
              index2 = var_index_list[v2]
              atom2 = re.split(v2,atom1[1]) #Sub
              #print(index2)
              #print(atom2)         
              #print(atom2[1])
              ops_list.append(atom2[1])
              if atom2[1] == 'Add':
                data_out = data_out + data[:,index2]
              elif atom2[1] == 'Sub':
                data_out = data_out - data[:,index2]
              elif atom2[1] == 'Prod':
                data_out = data_out * data[:,index2]
                #data_out = [a * b for a, b in zip(data_out, data[:,index2])]
                                    

  loss_vector = abs(y - data_out)
  loss_sum = sum(loss_vector)
  #print(loss_sum)
  #print(len(loss_vector))
  #print(len(y))
  loss_avg =sum(loss_vector)/len(loss_vector)
  stringfunc = printFunction(var_list, ops_list, tran_list)
  return loss_avg, data_out, stringfunc


def createDictOfPredTerms(names, kb_preds, op_preds):
    '''
    create dicts for the transformation preds and operaiton preds
    pred names act as keys for numeric values (intially zero) which
    will be used to count the number of present preds when looking at
    the final return rules from a dNL unit
    params:
    names: list of variable names [X1, X2]
    kb_preds: list of variable transformation presd ['Sine', 'Square, ...]
    op_preds: list of operation predicates ['Add', 'Product', ...]
    '''
    pred_trans_name_list =names.copy()
    for v in names:
        for t in kb_preds:
            pred_name = v + t
            pred_trans_name_list.append(pred_name)

    pred_trans_dict = {x : 0 for x in pred_trans_name_list}

    pred_ops_name_list =[]
    empty=''
    front = empty.join(names)
    rev = names[::-1]
    reverse = empty.join(rev)
    for t in op_preds:     
        if t != 'Sub':
            pred_obs = empty.join([front,t])
            pred_ops_name_list.append(pred_obs)
        else:
            pred_obs = empty.join([front, t])
            pred_ops_name_list.append(pred_obs)

            pred_obs = empty.join([reverse, t])
            pred_ops_name_list.append(pred_obs)


    pred_ops_dict = {x : 0 for x in pred_ops_name_list}

    return pred_trans_dict, pred_ops_dict, pred_trans_name_list, pred_ops_name_list



def buildCntKnowledgeBase(bin, names, num_cont_intervals, data_x, data_y, phase):
    TEST_SET_INDEX=0
    #we need the max/min array for potential input feature values
    #this will be used to determine the interval predicate values (predX < a)
    max_init=np.max(data_x,axis=0)
    min_init=np.min(data_x,axis=0)

    #Here we create vector of permutations up to 200, used for selecting random
    #indicies in the origional dataset, used for training the dNL model
    #How we define K also determines the Batch Size for the continous input
    K = 5 #K for K-fold validation
    L = data_x.shape[0] // K 
    inds = np.random.permutation( L*K) #perms of 200; [34, 55, 3, ...]
    DataSets=[]


    #So we iterate 5 times and collect data instances based on the indicies 
    for i in range(K):
        #we recreate the dataset using the random perm indicies
        #this line just creates a randomly shuffled dataset, theres probably a scikit method that would do this...
        DataSets.append(  (data_x[inds[ i*L:(i+1)*L],:] , data_y[inds[ i*L:(i+1)*L]]))

    #define predicates
    Constants = dict({})
    #PredCollection is a very important datastructure for dNL
    #will eventaully contain the KB and objecrive predicates
    predColl = PredCollection (Constants)
    var_count = len(names)
    #my implementation for continous predicates are nonlinear(nl), operation(add, product)
    #add_continuos provides the  predicates of the form (X1>A)
    #add_continuous_nl provides non-linear transformations on the feature values (sin,cos, exp)
    #a number of non-linear predicates are created for each variable
    for i in range(var_count):
        predColl.add_continuous(name = names[i], no_lt = num_cont_intervals, no_gt = num_cont_intervals, max_init = max_init[i], min_init = min_init[i])

    for i in range(bin):
        predColl.add_pred(name='class_%d'%(i+1),arguments=[] , variables=[] , pFunc = DNF(phase + '_' +'class_%d'%(i+1),terms=2,init=[-1,.1,-1,.1],sig=2)   ,use_cnt_vars=True,inc_preds=[])

    predColl.initialize_predicates()

    #add background facts to the KB
    bg_train = []
    bg_test = []

    for j in range(K):
            for i in range(L):
                bg = Background( predColl)

                for b in range(bin):
                    bg.add_example(pred_name='class_%d'%(b+1),pair=( ), value= float(DataSets[j][1][i]==b) )
                #forloop adds...????
                for k in range(var_count):
                    bg.add_continuous_value( names[k],( (DataSets[j][0][i,k] ,),))
                if j == TEST_SET_INDEX:
                    bg_test.append(bg)
                else:
                    bg_train.append(bg)
    BS  = len(bg_test)

    return predColl, bg_train, bg_test, BS, DataSets, L

                







def buildKnowledgeBase(bin, names, num_cont_intervals, kb_preds, op_preds, data_x, data_y, transform=True, operation=False , phase='', var_smt = False, subbothfirst = [True,True]):
    '''
    method responsible for building the targert predicates and background predicates
    for which the dNL unit will eventually use to learn rules defining the non-linear
    relaitonship in the data
    '''
    # for 5-fold we should run the program 5 times with TEST_SET_INDEX from 0 to 4
    TEST_SET_INDEX=0
    #we need the max/min array for potential input feature values
    #this will be used to determine the interval predicate values (predX < a)
    max_init=np.max(data_x,axis=0)
    min_init=np.min(data_x,axis=0)

    both = subbothfirst[0]
    first = subbothfirst[1]

    #Here we create vector of permutations up to 200, used for selecting random
    #indicies in the origional dataset, used for training the dNL model
    #How we define K also determines the Batch Size for the continous input
    K = 5 #K for K-fold validation
    L = data_x.shape[0] // K 
    inds = np.random.permutation( L*K) #perms of 200; [34, 55, 3, ...]
    DataSets=[]

    #So we iterate 5 times and collect data instances based on the indicies 
    for i in range(K):
        #we recreate the dataset using the random perm indicies
        #this line just creates a randomly shuffled dataset, theres probably a scikit method that would do this...
        DataSets.append(  (data_x[inds[ i*L:(i+1)*L],:] , data_y[inds[ i*L:(i+1)*L]]))

    #define predicates
    Constants = dict({})
    #PredCollection is a very important datastructure for dNL
    #will eventaully contain the KB and objecrive predicates
    predColl = PredCollection (Constants)
    var_count = len(names)
    #my implementation for continous predicates are nonlinear(nl), operation(add, product)
    #add_continuos provides the  predicates of the form (X1>A)
    #add_continuous_nl provides non-linear transformations on the feature values (sin,cos, exp)
    #a number of non-linear predicates are created for each variable
    if var_smt:
        predColl.add_continuous(name = names[i], no_lt = num_cont_intervals, no_gt = num_cont_intervals, max_init = max_init[i], min_init = min_init[i])

    
    
    if transform:
        for i in range(var_count):
            if var_smt:
                predColl.add_continuous(name = names[i], no_lt = num_cont_intervals, no_gt = num_cont_intervals, max_init = max_init[i], min_init = min_init[i])
            for t in kb_preds:
                print(t)
                if t == 'Sine':
                    predColl.add_continuous_sin(name = names[i], no_lt = num_cont_intervals, no_gt = num_cont_intervals, max_init = max_init[i], min_init = min_init[i])
                elif t == 'Exp':
                    predColl.add_continuous_exp(name = names[i], no_lt = num_cont_intervals, no_gt = num_cont_intervals, max_init = max_init[i], min_init = min_init[i])
                elif t == 'Square':
                    predColl.add_continuous_square(name = names[i], no_lt = num_cont_intervals, no_gt = num_cont_intervals, max_init = max_init[i], min_init = min_init[i])

        #add class target predicates we are trying to learn
        # the 3 refers to the number of discretized classes we are using, should be modular in the future
        #disjunctive neurons(DNF) are used to define our class predicates
        #cascading a conjunction layer with one disjunctive neuron we can form a dNL-DNF construct FIX
        for i in range(bin):
            predColl.add_pred(name=phase + '_' + 'class_%d'%(i+1),arguments=[] , variables=[] , pFunc = DNF(phase + '_' +'class_%d'%(i+1),terms=2,init=[-1,.1,-1,.1],sig=2)   ,use_cnt_vars=True,inc_preds=[])

        predColl.initialize_predicates()

        #add background facts to the KB
        bg_train = []
        bg_test = []

        for j in range(K):
            for i in range(L):
                bg = Background( predColl)
                for b in range(bin):
                    bg.add_example(pred_name=phase + '_' +'class_%d'%(b+1),pair=( ), value= float(DataSets[j][1][i]==b) )
                #forloop adds...????
                for k in range(var_count):
                    if var_smt:
                        bg.add_continuous_value( names[k],( (DataSets[j][0][i,k] ,),))
                    #bg.add_continuous_value_nl( names[k],( (DataSets[j][0][i,k] ,),))
                    for t in kb_preds:
                        print(t)
                        if t =='Sine':
                            bg.add_continuous_value_sin( names[k],( (DataSets[j][0][i,k] ,),))
                        elif t == 'Exp':
                            bg.add_continuous_value_exp( names[k],( (DataSets[j][0][i,k] ,),))
                        elif t == 'Square':
                            bg.add_continuous_value_square( names[k],( (DataSets[j][0][i,k] ,),))

                if j == TEST_SET_INDEX:
                    bg_test.append(bg)
                else:
                    bg_train.append(bg)
        #Here we formally define the batch size
        BS  = len(bg_test)

    elif operation:
        j=1
        for i in range(var_count):
            #IDEA: the layering of the operation predicates should be layered after transformation on the single variable
            for t in op_preds:
                print(t)
                if t =='Add':
                    predColl.add_continuous_add(name1 = names[i], name2=names[j], no_lt= num_cont_intervals, no_gt = num_cont_intervals, max_init1 =  max_init[i], min_init1 = min_init[i], max_init2=max_init[j], min_init2 = min_init[j] )
                elif t == 'Prod':
                    predColl.add_continuous_prod(name1 = names[i], name2=names[j], no_lt= num_cont_intervals, no_gt = num_cont_intervals, max_init1 =  max_init[i], min_init1 = min_init[i], max_init2=max_init[j], min_init2 = min_init[j] )
                elif t == 'Sub':
                    predColl.add_continuous_sub(name1 = names[i], name2=names[j], no_lt= num_cont_intervals, no_gt = num_cont_intervals, max_init1 =  max_init[i], min_init1 = min_init[i], max_init2=max_init[j], min_init2 = min_init[j],both =both,first = first )

            j += 1
            if j == var_count:
                break
        #add class target predicates we are trying to learn
        # the 3 refers to the number of discretized classes we are using, should be modular in the future
        #disjunctive neurons(DNF) are used to define our class predicates
        #cascading a conjunction layer with one disjunctive neuron we can form a dNL-DNF construct FIX
        for i in range(bin):
            predColl.add_pred(name=phase + '_' +'class_%d'%(i+1),arguments=[] , variables=[] , pFunc = DNF(phase + '_' +'class_%d'%(i+1),terms=2,init=[-1,.1,-1,.1],sig=2)   ,use_cnt_vars=True,inc_preds=[])

        predColl.initialize_predicates()

        #add background facts to the KB
        bg_train = []
        bg_test = []

        for j in range(K):
            for i in range(L):
                bg = Background( predColl)

                for b in range(bin):
                    bg.add_example(pred_name=phase + '_' +'class_%d'%(b+1),pair=( ), value= float(DataSets[j][1][i]==b) )
                #forloop adds...????
                for k in range(var_count):

                    for t in op_preds:
                        print(t)
                        if t =='Add':
                            bg.add_continuous_value_add(names[k], names[k+1], ( (DataSets[j][0][i,k] ,),),( (DataSets[j][0][i,k + 1] ,), ))
                        elif t == 'Prod':
                            bg.add_continuous_value_prod(names[k], names[k+1], ( (DataSets[j][0][i,k] ,),),( (DataSets[j][0][i,k + 1] ,), ))
                        elif t == 'Sub':
                            bg.add_continuous_value_sub(names[k], names[k+1], ( (DataSets[j][0][i,k] ,),),( (DataSets[j][0][i,k + 1] ,), ),both=both,first=first)
                    if (k+1) == (var_count - 1):
                            break

                if j == TEST_SET_INDEX:
                    bg_test.append(bg)
                else:
                    bg_train.append(bg)
        BS  = len(bg_test)

    else: #KB has all operation and transformation preds
        print(0)
        for i in range(var_count):
            predColl.add_continuous(name = names[i], no_lt = num_cont_intervals, no_gt = num_cont_intervals, max_init = max_init[i], min_init = min_init[i])
            for t in kb_preds:
                print(t)
                if t == 'Sine':
                    predColl.add_continuous_sin(name = names[i], no_lt = num_cont_intervals, no_gt = num_cont_intervals, max_init = max_init[i], min_init = min_init[i])
                elif t == 'Exp':
                    predColl.add_continuous_exp(name = names[i], no_lt = num_cont_intervals, no_gt = num_cont_intervals, max_init = max_init[i], min_init = min_init[i])
                elif t == 'Square':
                    predColl.add_continuous_square(name = names[i], no_lt = num_cont_intervals, no_gt = num_cont_intervals, max_init = max_init[i], min_init = min_init[i])

        for i in range(var_count):
            #IDEA: the layering of the operation predicates should be layered after transformation on the single variable
            
            predColl.add_continuous_add(name1 = names[i], name2=names[j], no_lt= num_cont_intervals, no_gt = num_cont_intervals, max_init1 =  max_init[i], min_init1 = min_init[i], max_init2=max_init[j], min_init2 = min_init[j] )
            predColl.add_continuous_sub(name1 = names[i], name2=names[j], no_lt= num_cont_intervals, no_gt = num_cont_intervals, max_init1 =  max_init[i], min_init1 = min_init[i], max_init2=max_init[j], min_init2 = min_init[j] )
            predColl.add_continuous_prod(name1 = names[i], name2=names[j], no_lt= num_cont_intervals, no_gt = num_cont_intervals, max_init1 =  max_init[i], min_init1 = min_init[i], max_init2=max_init[j], min_init2 = min_init[j] )

        
            j += 1
            if j == var_count:
                break
        #add class target predicates we are trying to learn
        # the 3 refers to the number of discretized classes we are using, should be modular in the future
        #disjunctive neurons(DNF) are used to define our class predicates
        #cascading a conjunction layer with one disjunctive neuron we can form a dNL-DNF construct FIX
        for i in range(bin):
            predColl.add_pred(name=phase + '_' +'class_%d'%(i+1),arguments=[] , variables=[] , pFunc = DNF(phase + '_' +'class_%d'%(i+1),terms=2,init=[-1,.1,-1,.1],sig=2)   ,use_cnt_vars=True,inc_preds=[])

        predColl.initialize_predicates()

        #add background facts to the KB
        bg_train = []
        bg_test = []

        for j in range(K):
            for i in range(L):
                bg = Background( predColl)
                for b in range(bin):
                    bg.add_example(pred_name=phase + '_' +'class_%d'%(b+1),pair=( ), value= float(DataSets[j][1][i]==b) )
                #forloop adds...????
                for k in range(var_count):
                    bg.add_continuous_value( names[k],( (DataSets[j][0][i,k] ,),))
                    #bg.add_continuous_value_nl( names[k],( (DataSets[j][0][i,k] ,),))
                    for t in kb_preds:
                        print(t)
                        if t =='Sine':
                            bg.add_continuous_value_sin( names[k],( (DataSets[j][0][i,k] ,),))
                        elif t == 'Exp':
                            bg.add_continuous_value_exp( names[k],( (DataSets[j][0][i,k] ,),))
                        elif t == 'Square':
                            bg.add_continuous_value_square( names[k],( (DataSets[j][0][i,k] ,),))

                #forloop ads operation predicates for vars
                for k in range(var_count):

                    for t in op_preds:
                        print(t)
                        if t =='Add':
                            bg.add_continuous_value_add(names[k], names[k+1], ( (DataSets[j][0][i,k] ,),),( (DataSets[j][0][i,k + 1] ,), ))
                        elif t == 'Prod':
                            bg.add_continuous_value_prod(names[k], names[k+1], ( (DataSets[j][0][i,k] ,),),( (DataSets[j][0][i,k + 1] ,), ))
                        elif t == 'Sub':
                            bg.add_continuous_value_sub(names[k], names[k+1], ( (DataSets[j][0][i,k] ,),),( (DataSets[j][0][i,k + 1] ,), ),both=both,first=first)
                    if (k+1) == (var_count - 1):
                            break
                if j == TEST_SET_INDEX:
                    bg_test.append(bg)
                else:
                    bg_train.append(bg)
        BS  = len(bg_test)            


    return predColl, bg_train, bg_test, BS, DataSets, L


def constructFunction(predlist1, predlist2, data_y_cnt, data_x_cnt):

    pass
