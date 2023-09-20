from numpy import int64
from Library.NLILPRLEngine import *
import argparse
from Library.mylibw import read_by_tokens
from Library.DNF import DNF
from Library.CNF import CNF
from Library.PredicateLibV5 import PredFunc
from Library.dataParse import *
from Library.dNLunit import *
from sklearn.metrics import accuracy_score, precision_recall_curve,auc,precision_recall_fscore_support
import operator
import scipy.signal
import time
import re




class doubleUnit():
    '''
    class is responsible for running seperate dNL units and storing Kb
    this unit will run either transformation/operation dNL learning or both
    '''
    def __init__(self, names, bin, num_cont_intervals, TEST_SET_INDEX, data_x, data_y_cnt, data_y_dsc, kb_preds, op_preds, phases, var_smt = False, train_transform = True, subbothfirst=[True,True], iterations=100*12):
        self.names = names
        self.kb_preds = kb_preds
        self.op_preds = op_preds
        self.data_x = data_x
        self.data_y_cnt = data_y_cnt
        self.data_y_dsc = data_y_dsc
        self.phase1 = phases[0]
        self.phase2 = phases[1]
        self.bin = bin
        self.num_cont_intervals = num_cont_intervals
        self.TEST_SET_INDEX = TEST_SET_INDEX
        #self.L = L
        self.var_smt = var_smt
        self.data_x_tran = None
        self.train_transform = train_transform
        self.subbothfirst = subbothfirst

        self.iterations = iterations

    def runBasicUnit(self):


        pred_trans_dict, pred_ops_dict, pred_trans_name_list, pred_ops_name_list = createDictOfPredTerms(self.names, self.kb_preds, self.op_preds)
        predColl, bg_train, bg_test, BS, DataSets, L = buildKnowledgeBase(self.bin, self.names.copy(), self.num_cont_intervals, self.kb_preds, self.op_preds, self.data_x, self.data_y_dsc, transform=True, operation=False, phase= self.phase1, var_smt=self.var_smt, subbothfirst=self.subbothfirst )
        transformation_learner = dNLunit(self.bin, bg_train, bg_test, DataSets,self.TEST_SET_INDEX, L, self.num_cont_intervals, self.names.copy(), pred_trans_name_list, predColl, self.phase1, self.iterations)
        predlist,  highest_acc, ruleDefListTran = transformation_learner.rundNLunit(BS)
        print(predlist)
        print(highest_acc)
        return  predlist, highest_acc, ruleDefListTran



    def runUnit(self):
        '''
        primary method for running dNL units for transformations and operations
        '''
        predlist = None
        predlist2 = None
        highest_acc = 0
        highest_acc2 =0
        ruleDefListOps = []
        ruleDefListTran = []
        #we indicate here that we want to train a dNL to identify transformation units
        if self.train_transform:
            pred_trans_dict, pred_ops_dict, pred_trans_name_list, pred_ops_name_list = createDictOfPredTerms(self.names, self.kb_preds, self.op_preds)

            predColl, bg_train, bg_test, BS, DataSets, L = buildKnowledgeBase(self.bin, self.names.copy(), self.num_cont_intervals, self.kb_preds, self.op_preds, self.data_x, self.data_y_dsc, transform=True, operation=False, phase= self.phase1, var_smt=self.var_smt, subbothfirst=self.subbothfirst )
            transformation_learner = dNLunit(self.bin, bg_train, bg_test, DataSets,self.TEST_SET_INDEX, L, self.num_cont_intervals, self.names.copy(), pred_trans_name_list, predColl, self.phase1, self.iterations)
            predlist,  highest_acc, ruleDefListTran = transformation_learner.rundNLunit(BS)
            print(predlist)
            transformation_learner.retrainModel()

        #If the number vraibles is more than one, than we can learn an operation
        #between variables
        if len(self.names) > 1:
            #here we discretize the  target output 
            _ , self.data_x_tran = transformDataset(self.data_y_cnt.copy(), self.data_x.copy(), list(predlist),self.names)

            #data_x = data_x_tran #data[:,1:] #(200,2)
            #data_y = data_y_dsc #data[:,0]  #(200,)

            #in some cases with extreme values we may want to normalize
            data_xm = np.mean( self.data_x_tran, axis=0,keepdims=True)
            data_xv = np.std( self.data_x_tran, axis=0,keepdims=True)
            #data_x = (data_x-data_xm) / data_xv
            #data_x
            predColl2, bg_train2, bg_test2, BS2, DataSets2, L2 = buildKnowledgeBase(self.bin, self.names.copy(), self.num_cont_intervals, self.kb_preds, self.op_preds, self.data_x_tran, self.data_y_dsc, transform=False, operation=True, phase= self.phase2, var_smt=self.var_smt, subbothfirst=self.subbothfirst )
            operation_learner = dNLunit(self.bin, bg_train2, bg_test2, DataSets2, self.TEST_SET_INDEX, L2, self.num_cont_intervals, self.names.copy(), pred_ops_name_list, predColl2, self.phase2, self.iterations)
            predlist2, highest_acc2, ruleDefListOps = operation_learner.rundNLunit(BS2)
            print(predlist)
            print(highest_acc)
            print(predlist2)
            print(highest_acc2)
            operation_learner.retrainModel()
        
        return predlist2, predlist, highest_acc2, highest_acc, ruleDefListOps, ruleDefListTran

    def runOperationUnit(self, predlist):
        '''
        here we train a dNL unit but only only the operation predicates
        the data is transformed based on the transformation preds in predlist
        and then the operation predicates are learned
        '''
        pred_trans_dict, pred_ops_dict, pred_trans_name_list, pred_ops_name_list = createDictOfPredTerms(self.names, self.kb_preds, self.op_preds)
        _ , self.data_x_tran = transformDataset(self.data_y_cnt.copy(), self.data_x.copy(), list(predlist),self.names)
        predColl2, bg_train2, bg_test2, BS2, DataSets2, L2 = buildKnowledgeBase(self.bin, self.names.copy(), self.num_cont_intervals, self.kb_preds, self.op_preds, self.data_x_tran, self.data_y_dsc, transform=False, operation=True, phase= self.phase2, var_smt=self.var_smt, subbothfirst=self.subbothfirst )
        operation_learner = dNLunit(self.bin, bg_train2, bg_test2, DataSets2, self.TEST_SET_INDEX, L2, self.num_cont_intervals, self.names.copy(), pred_ops_name_list, predColl2, self.phase2)
        predlist2, highest_acc2, ruleDefListOps = operation_learner.rundNLunit(BS2)
        print(predlist2)
        print(highest_acc2)
        operation_learner.retrainModel()
        return predlist2, highest_acc2, ruleDefListOps






    def lossOnFunction(self, predlist2, predlist):
        '''
        takes the learned non-linear function and calculates the loss on the 
        real-valued target y
        params:
        predlist2: list of the operation predicates
        predlist: list of the transformation predicates
        '''
        _ , data_x_tran = transformDataset(self.data_y_cnt.copy(), self.data_x.copy(), list(predlist),self.names) #predlist2 is empty in some cases, where predlist=[X1EXP,X2EXP]
        loss, other, stringfunc = operationOrder(self.data_y_cnt.copy(), self.data_x_tran.copy(), predlist2, predlist, self.names) #sometimes predList has ONE transformation
        print(loss)
        print(other)
        return loss, stringfunc
    
def splitData(names, data_x):
    '''
    Split a dataset into vector of seperate variables
    '''
    var_length = len(names)
    list_vars = []
    for i in range(var_length):
        print("no")
        temp = np.reshape(data_x[:,i].copy(),(len(data_x[:,i]),1))
        list_vars.append(temp)
    return list_vars

def reduceTransformationList(var_switch, worst_var, flat_pred_list, names, temp_kb_preds, temp_removal_list):
    var_switch[worst_var] = True #indicate that we will augent the kb for the corresponding pred
    temp_pred_to_remove = flat_pred_list[worst_var] 
    for e in names:
        if e in temp_pred_to_remove:
            temp_pred_to_remove = temp_pred_to_remove.replace(e, "")
    #print(temp_pred_to_remove)
    for i in range(len(temp_kb_preds[worst_var])):
        if  temp_kb_preds[worst_var][i] != temp_pred_to_remove:
            temp_removal_list.append( temp_kb_preds[worst_var][i]) #kb_preds[i]
            temp_removal_list = list(dict.fromkeys(temp_removal_list))
    

    return var_switch, worst_var, flat_pred_list, names, temp_kb_preds, temp_removal_list


def reduceOperationList():
    pass

if __name__ == "__main__":

    #TEST ON DATA BETWEEN 1-10 continous

   
    #DATA_FILE = './data/continuous/CNTComplexSinX1AddPowerX2.data' #SUCCESS

    
    #DATA_FILE = './data/continuous/CNTComplexSimpX3sinSquare.data'  #sinX1 aDD SQUAREx2...SUCCESS...CALC CORRECT (din't normalize)
    
    #DATA_FILE = './data/continuous/yacht_hydrodynamics.data'
    DATA_FILE =  'dNL-NL/data/continuous/yachthydrodynamics.data'
    
    #NOTE: normalizing this simpX3sinsquare does not help, seems to do better with OG values
    #NOTE: (normalize by 5)...fails to cal the true loss
    #DATA_FILE ='./data/continuous/fX3.data' #powerX(x[:,0]) - expX(x[:,1]) SUCCESS...BUT CALCULATED WRONG FIX TRUELOSS
    #NOTE: Can solve with OG values but will break if 'X1Exp' OPERATIONS 'X1Exp'
    #ISSUE: The Sub selection is not working properly.
    #ISSUE: when temp_op_preds = ['Sub'], I presume X1X2Sub, the dNL cannot identify the rule for some reason.
    #       this means i need to check the BK and predicates for when X1X2Sub
    #DATA_FILE ='./data/continuous/fX4.data' #expX(x[:,0]) * sinX(x[:,1]) SUCCESS...CALC CORRECT
    #NOTE: Can solve with OG values
    #DATA_FILE = './data/continuous/fX5.data'    #powerX(x[:,0]) - sinX(x[:,1]) #SUCCESS...CALC CORRECT
    #NOTE: Can solve with OG values
    # for 5-fold we should run the program 5 times with TEST_SET_INDEX from 0 to 4

    #fx3 -1.202460284164776793e+06 12 14 (144 - 1202604.284164)
    #ISSUE: the continous preds for X1X2Sub are very incorrect as for max/min values
    
    
    TEST_SET_INDEX=0
    #get data from text file and seperate input/ouput
    data_cnt = np.genfromtxt(DATA_FILE, delimiter='')#(200,3)
    #data_cnt = data_cnt*50 #to get larger random values
    data_x_cnt = data_cnt[:,0:-1]
    
    data_y_cnt = data_cnt[:,-1]
    #collect continous data file\
    #input, number of bins to discretize our output
    size = data_y_cnt.shape[0]
    bin = 3
    #number of cnt intervals for a given cnt feature
    num_cont_intervals = 10
    #list of feature names used to create predicate names, and the number of feature variables
    names=['Buoyancy', 'Prismatic' , 'Displacement', 'Beamdraught', 'Lengthbeam', 'Froude']
    names1 = [names[0]]
    names2 = [names[1]]
    names3 = [names[2]]
    names4 = [names[3]]
    names5 = [names[4]]
    names6 = [names[5]]
    #pred_trans_name_list =names.copy()
    #names = ['X1']
    var_count = 6 #variable count specific to the current dataset
    #list of function predicates
    kb_preds = ['Exp', 'Square', 'Sine']
    #kb_preds = ['Exp', 'Square']
    kb_preds_lim = ['Sine','']
    temp_kb_preds = [kb_preds.copy()]*var_count #create copy list for when we remove preds

    #list of operation predicates
    op_preds = ['Add', 'Prod', 'Sub']
    #op_preds = ['Sub']
    temp_op_preds = op_preds.copy()
    #op_preds = ('Add','Prod')
    phase1 = "STAGE1"
    phase2 = "STAGE2"
    phases = (phase1,phase2)
    #here we discretize the  target output 
    #data_x = (data_x-data_xm) / data_xv
    #data_xm = np.mean( data_x_cnt,axis=0,keepdims=True)
    #data_xv = np.std( data_x_cnt,axis=0,keepdims=True)
    #data_x_cnt = (data_x_cnt-data_xm) / data_xv
    #data_ym = np.mean( data_y_cnt,axis=0,keepdims=True)
    #data_yv = np.std( data_y_cnt,axis=0,keepdims=True)
    #data_y_cnt = (data_y_cnt - data_ym)/data_yv

    #data_y_dsc, data_x_cnt = discreteBetterClassifier(data_y_cnt, data_x_cnt, bin, size)
    data_y_dsc, data_x_cnt = discreteEqualFreqClassifier(data_y_cnt,data_x_cnt,bin,size)

    #zeroUnit = doubleUnit(names, bin, num_cont_intervals, TEST_SET_INDEX, data_x_cnt, data_y_cnt, data_y_dsc, kb_preds, op_preds, phases, var_smt = False)
    #predOPS, predTRAN = zeroUnit.runUnit()
    #loss = zeroUnit.lossOnFunction(predOPS,predTRAN)

    avg_loss = 100
    whole_flag = False
    var_length = len(names)
    #var_kb_agument = [True]*var_length
    var_list = splitData(names, data_x_cnt)
    perm_kb_preds = False
    episode = 0 #number of episodes
    temp_removal_list = [] #list for removing specific transformation preds from kb_pred list
    var_switch = [False] * var_length
    time_start = time.time()
    extraced_rule_list = [] #store true loss and extraxted rules.
    extraced_rule_set = []
    subbothfirst=[True,True]
    extracted_base_rule_set = []

    
    
    if True:
        zeroUnit = doubleUnit(names, bin, num_cont_intervals, TEST_SET_INDEX, data_x_cnt.copy(), data_y_cnt.copy(), data_y_dsc.copy(), kb_preds, op_preds, phases, var_smt = False,subbothfirst=subbothfirst)
        predlist, highest_acc, ruleDefListTran = zeroUnit.runBasicUnit()

        #loss, stringfunc = zeroUnit.lossOnFunction(predOPS,predTRAN) 
    else:
    
    
        while avg_loss >  0.05:
            #first try to learn NL function on the whole dataset
            #TEMP: var_switch[True, True] and temp_kb_preds = [['Power'],['Exp']]#########
            #var_switch = [True] * var_length
            #temp_kb_preds = [['Exp'],['Exp']]
            #TEMP#########################################################################
            if whole_flag:
                zeroUnit = doubleUnit(names, bin, num_cont_intervals, TEST_SET_INDEX, data_x_cnt.copy(), data_y_cnt.copy(), data_y_dsc.copy(), kb_preds, op_preds, phases, var_smt = False,subbothfirst=subbothfirst)
                predOPS, predTRAN, accOPS, accTRAN, ruleOps, ruleTran = zeroUnit.runUnit()

                loss, stringfunc = zeroUnit.lossOnFunction(predOPS,predTRAN)
                whole_flag = False
                if loss < 0.05:
                    break
                else:
                    avg_loss = loss
            else:
                
                pred_set = set()
                pred_list = []
                full_pred_name_list = []
                #learn transformatons on each variable seperatly 
                var_rule_list =[]
                for i in range(var_length):
                    names1 = [names[i]]
                    var_data = var_list[i]
                    if var_switch[i]: #if this is true then use an augmented transformation pred list kb_preds
                        firstUnit = doubleUnit(names1, bin, num_cont_intervals, TEST_SET_INDEX, var_data.copy(), data_y_cnt.copy(), data_y_dsc.copy(), temp_kb_preds[i], op_preds, phases, var_smt = False)
                        var_switch[i] = False
                        if len(temp_kb_preds[i]) == 0:
                            temp_kb_preds = ['Exp', 'Square', 'Sine']
                    else:
                        firstUnit = doubleUnit(names1, bin, num_cont_intervals, TEST_SET_INDEX, var_data.copy(), data_y_cnt.copy(), data_y_dsc.copy(), kb_preds, op_preds, phases, var_smt = False)
                    
                    _, predTRANone, accOPS, accTRAN, ruleOps, ruleTran  = firstUnit.runUnit()
                    var_rule_list.append([ruleTran, accTRAN])
                    pred_list.append(predTRANone) #save preds learned for each variable
                    full_pred_name_list.append(predTRANone)
                
                #CAS1: If transformations used before, then don't use same opertation again and remove specified op
                flat_pred_list = [item for sublist in pred_list for item in sublist]
                all_operation_flag = True
                if len(extraced_rule_set) > 0: #do we have prexisting rules found
                    for idx, rule in enumerate(extraced_rule_set[:]): 
                        rule = rule[0]
                        res = [(ele[0],ele2[0]) for ele in pred_list for ele2 in pred_list if ele[0] != ele2[0] and (ele[0] in rule and ele2[0] in rule)]
                        if len(res) > 0:
                            #remove operation for ops_preds
                            clean_rule = rule.strip()
                            rule_list = list(clean_rule.split(" "))
                            old_ops = [word for word in rule_list if word not in flat_pred_list] #['X1Exp','Prod','X2Square0'] [['X1Exp'],['X2Square']]
                            old_ops = old_ops[0]
                            if old_ops in temp_op_preds:
                                if old_ops == 'Sub': #As the Sub operation includes order of operations between vars we need to clarify which to remove
                                    ruleSub = extraced_rule_set[idx][1]
                                    bothlocal = False
                                    ValInts = list(map(int, re.findall(r'\d+', ruleSub)))
                                    if ValInts[0] < ValInts[1]:
                                        firstllocal = False
                                    else:
                                        firstllocal = True
                                    subbothfirst[0] =bothlocal
                                    subbothfirst[1] = firstllocal
                                    all_operation_flag = False

                                else:
                                    temp_op_preds.remove(old_ops)
                                    all_operation_flag = False
                            
                            if len(temp_op_preds) == 0:
                                temp_op_preds = op_preds.copy()
                                all_operation_flag = True
                #TEMP:
                #all_operation_flag = False
                if not all_operation_flag:
                    if len(temp_op_preds) == 0:
                        temp_op_preds = op_preds.copy()
                    #TEMP: TEST 1 3##################################################
                    #subbothfirst = [False, True]
                    #TEMP: TEST 1 3##################################################
                    thirdUnit = doubleUnit(names, bin, num_cont_intervals, TEST_SET_INDEX, data_x_cnt.copy(), data_y_cnt.copy(), data_y_dsc.copy(), kb_preds, temp_op_preds, phases, var_smt = False, train_transform = False, subbothfirst=subbothfirst, iterations=100*60)
                    #create single dim list of the transformation preds
                    flat_pred_list = [item for sublist in pred_list for item in sublist]
                    predOPStwo, acc3, ruleTrueOps = thirdUnit.runOperationUnit( flat_pred_list.copy())
                    loss, stringfunc = thirdUnit.lossOnFunction(predOPStwo.copy(),flat_pred_list.copy())
                    #break if nl function matches data
                    extraced_rule_set.append([stringfunc,predOPStwo[0]])
                    extraced_rule_list.append([loss, stringfunc])
                    full_pred_name_list.append(predOPStwo[0])
                    extracted_base_rule_set.append(full_pred_name_list)
                    subbothfirst = [True,True]
                    temp_op_preds = op_preds.copy()

                else:
                    #TEMP: TEST 1 ###############################################################
                    #subbothfirst = [False, True]
                    #TEMP: TEST 1 ###############################################################
                    thirdUnit = doubleUnit(names, bin, num_cont_intervals, TEST_SET_INDEX, data_x_cnt.copy(), data_y_cnt.copy(), data_y_dsc.copy(), kb_preds, op_preds, phases, var_smt = False, train_transform = False, subbothfirst=subbothfirst, iterations=100*60)
                    #create single dim list of the transformation preds
                    flat_pred_list = [item for sublist in pred_list for item in sublist]
                    #ISSUE: predOPStwo is [''] when predlist = ['X1Exp','X2Exp']
                    predOPStwo, acc3, ruleTrueOps = thirdUnit.runOperationUnit( flat_pred_list.copy())
                    loss, stringfunc = thirdUnit.lossOnFunction(predOPStwo.copy(),flat_pred_list.copy()) #error when nothing in flat_pred_list
                    #break if nl function matches data
                    extraced_rule_set.append([stringfunc, predOPStwo[0]])
                    extraced_rule_list.append([loss, stringfunc])
                    full_pred_name_list.append(predOPStwo[0])
                    extracted_base_rule_set.append(full_pred_name_list)
                    subbothfirst = [True,True]
                    temp_op_preds = op_preds.copy()

                all_operation_flag = True
                if loss < 0.05:
                    break
                else:

                    #CASE 1: trueloss is low/non-zero and accuracy is high for subset of vars, low for other var

                    #CASE 2: trueloss is high and accuracy is low for single var.
                    #remove transformation pred from lowest performing variable
                    array_var_rule_current = np.array(var_rule_list)
                    #length = array_var_rule_current.shape([-1])
                    length = np.shape(array_var_rule_current)[-1]
                    worst_acc =np.amin(array_var_rule_current[:,1])
                    worst_var = int(np.argmin(array_var_rule_current[:,1]))#get the lowest accuracy index, corresponds to worst performing variable/rule
                    
                    var_switch, worst_var, flat_pred_list, names, temp_kb_preds, temp_removal_list = reduceTransformationList(var_switch, worst_var, flat_pred_list, names, temp_kb_preds, temp_removal_list)
                    if len(temp_removal_list) == 0: # x1 is now empty, we should empty out X2
                        temp_kb_preds[worst_var] = kb_preds.copy()
                        var_switch[worst_var] = False
                        #array_var_rule_current = array_var_rule_current[array_var_rule_current[:,1] != worst_acc]
                        k = 2
                        idx = np.argpartition(array_var_rule_current[:,1], 1)
                        print(idx)
                        second_worst_var = idx[1]

                        var_switch, second_worst_var, flat_pred_list, names, temp_kb_preds, temp_removal_list = reduceTransformationList(var_switch, second_worst_var, flat_pred_list, names, temp_kb_preds, temp_removal_list)
                        temp_kb_preds[second_worst_var] = temp_removal_list
                    else:
                        temp_kb_preds[worst_var] = temp_removal_list
                    temp_removal_list=[]

                    
                    avg_loss = loss
                #break if we reach 10 episodes
                if episode == 10:
                    break
                else:
                    episode += 1

    time_end = time.time()
    time_total = time_end - time_start    
    print("Final NL functions")
    #print(stringfunc)
    print("Total time elapsed")
    print(time_total)
    print("Rule Output")
    print("Transformation Layer : ")
    '''
    i= 0
    for v in var_rule_list:
        print("Transformation Layer on " + names[i])
        for e in v[0]:
            print(e)
        print( names[i] + " Highest Transformation accuracy is ", v[1])
        i+=1
    print("Operation Layer: ")
    for e in ruleTrueOps:
        print(e)
    print("Highest Operation accuracy is ", acc3)
    
    #print(extraced_rule_list)
    for i in extraced_rule_list:
        print(i)

    for e in extraced_rule_set:
        print(e)
'''


