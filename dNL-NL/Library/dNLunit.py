from Library.NLILPRLEngine import *
from Library.ILPRLEngine import *
import argparse
from sklearn.metrics import accuracy_score

class dNLunit():

    def __init__(self,bin, bg_train, bg_test,DataSets,TEST_SET_INDEX, L, num_cont_intervals, names, pred_trans_name_list, predColl, phase, iterations=100*12 ):
        x=0
        self.bin = bin
        self.bg_train = bg_train
        self.bg_test = bg_test
        self.DataSets = DataSets
        self.TEST_SET_INDEX = TEST_SET_INDEX
        self.L = L

        self.num_cont_intervals = num_cont_intervals
        self.names = names
        self.pred_trans_name_list = pred_trans_name_list
        self.predColl = predColl
        self.phase = phase
        self.model = None
        self.iterations = 1000*6 #iterations #100*60
    
    def bgs(self, it, is_train):
        if is_train:
            n=it%4
            return self.bg_train[ n*self.L:(n+1)*self.L]
        else:
            return self.bg_test


 ###########################################################################

    def disp_fn(self, eng, it, session, cost, outp):
        '''
        This method
        eng: 
        it:  
        session: 
        cost:
        outp: 
        '''
        Y_true =[]
        Y_score=[]
        
        cl_list = []
        for b in range(self.bin):
            cl = outp[self.phase +'_'+ 'class_%d'%(b+1)]
            cl_list.append(cl)
        #cl1 = outp['class_1']
        #cl2 = outp['class_2']
        #cl3 = outp['class_3']
        #we get the accuracy of our rules on the test data
        for i in range(self.L):
            Y_true.append ( self.DataSets[self.TEST_SET_INDEX][1][i])
            #Y_score.append( np.argmax( [cl1[i][0] ,cl2[i][0], cl3[i][0] ]))
            Y_score.append(np.argmax( [cl[i][0] for cl in cl_list] ))
        
        acc = accuracy_score(Y_true, Y_score)
        print('***********************************')
        print('accuracy score = ',  acc)
        print('***********************************')

        return acc


    def rundNLunit(self, BS):

        parser = argparse.ArgumentParser()

        parser.add_argument('--CHECK_CONVERGENCE',default=0,help='Check for convergence',type=int)
        parser.add_argument('--SHOW_PRED_DETAILS',default=0,help='Print predicates definition details',type=int)

        parser.add_argument('--PRINTPRED',default=1,help='Print predicates',type=int)
        parser.add_argument('--SYNC',default=0,help='Use L2 instead of cross entropy',type=int)
        parser.add_argument('--L2LOSS',default=1,help='Use L2 instead of cross entropy',type=int)
        parser.add_argument('--BS',default=BS,help='Batch Size',type=int)
        parser.add_argument('--T',default=1,help='Number of forward chain',type=int)
        parser.add_argument('--LR_SC', default={ (-1000,2):.005 ,  (2,1e5):.0051} , help='Learning rate schedule',type=dict)
        parser.add_argument('--ITEM_REMOVE_ITER',default=10000 ,help='length period of each item removal',type=int)
        parser.add_argument('--MAXTERMS',default=3 ,help='Maximum number of terms in each clause',type=int)
        parser.add_argument('--L1',default=.01 ,help='Penalty for maxterm',type=float)
        parser.add_argument('--L2',default=.01 ,help='Penalty for distance from binary',type=float)
        parser.add_argument('--L3',default=.000 ,help='Penalty for distance from binary',type=float)
        parser.add_argument('--L4',default=1.0 ,help='Penalty for distance from binary',type=float)
        parser.add_argument('--ALLTIMESTAMP',default=0 ,help='Add loss for each timestamp',type=int)
        parser.add_argument('--LR', default=.003 , help='Base learning rate',type=float)
        parser.add_argument('--FILT_TH_MEAN', default=.2 , help='Fast convergence total loss threshold MEAN',type=float)
        parser.add_argument('--FILT_TH_MAX', default=.2 , help='Fast convergence total loss threshold MAX',type=float)
        parser.add_argument('--OPT_TH', default=.05 , help='Per value accuracy threshold',type=float)
        parser.add_argument('--PLOGENT', default=.50 , help='Crossentropy coefficient',type=float)
        parser.add_argument('--BETA1', default=.90 , help='ADAM Beta1',type=float)
        parser.add_argument('--BETA2', default=.999 , help='ADAM Beta2',type=float)
        parser.add_argument('--EPS', default=1e-6, help='ADAM Epsillon',type=float)
        parser.add_argument('--GPU', default=1, help='Use GPU',type=int)
        parser.add_argument('--ITER', default=self.iterations, help='Maximum number of iteration',type=int) #100*20
        parser.add_argument('--ITER2', default=100, help='Epoch',type=int)
        parser.add_argument('--LOGDIR', default='./logs/Logic', help='Log Dir',type=str)
        parser.add_argument('--TB', default=0, help='Use Tensorboard',type=int)
        parser.add_argument('--ADDGRAPH', default=1, help='Add graph to Tensorboard',type=int)
        parser.add_argument('--CLIP_NORM', default=0, help='Clip gradient',type=float)

        parser.add_argument('--PRINT_WEIGHTS',default=0,help='Print raw weights',type=int)
        parser.add_argument('--SEED',default=0,help='Random seed',type=int)
        parser.add_argument('--BINARAIZE', default=0 , help='Enable binrizing at fast convergence',type=int)
        parser.add_argument('--MAX_DISP_ITEMS', default=50 , help='Max number  of facts to display',type=int)
        parser.add_argument('--W_DISP_TH', default=.1 , help='Display Threshold for weights',type=int)
        parser.add_argument('--DISP_BATCH_VALUES',default=[],help='Batch Size',type=list)

        parser.add_argument('--CONT_INTERVALS', default=self.num_cont_intervals, help='Number of continuous ranges', type=int)
        parser.add_argument('--CONT_VARIABLES', default=self.names, help='Number variables', type=list)

        parser.add_argument('--ATOM_NAMES',default=self.pred_trans_name_list, help='Dict of predicate names', type=dict)



        args = parser.parse_args()

        print('displaying config setting...')
        for arg in vars(args):
            print( '{}-{}'.format(arg, getattr(args, arg)))

        self.model = NLILPRLEngine( args = args, predColl = self.predColl, bgs = self.bgs, disp_fn = self.disp_fn, phase = self.phase)
        predlist, highest_acc = self.model.train_model()
        ruleDefList = self.model.ruleDefintionStringList
        return predlist, highest_acc, ruleDefList

    def retrainModel(self):
        '''
        clear out NN params for model
        '''
        self.model.retrain_model()

class vanilladNLunit():

    def __init__(self,bin, bg_train, bg_test,DataSets,TEST_SET_INDEX, L, num_cont_intervals, names, predColl, phase, iterations=100*12 ):
        x=0
        self.bin = bin
        self.bg_train = bg_train
        self.bg_test = bg_test
        self.DataSets = DataSets
        self.TEST_SET_INDEX = TEST_SET_INDEX
        self.L = L

        self.num_cont_intervals = num_cont_intervals
        self.names = names
        #self.pred_trans_name_list = pred_trans_name_list
        self.predColl = predColl
        self.phase = phase
        self.model = None
        self.iterations = 1000*6 #iterations #100*60
    
    def bgs(self, it, is_train):
        if is_train:
            n=it%4
            return self.bg_train[ n*self.L:(n+1)*self.L]
        else:
            return self.bg_test


 ###########################################################################

    def disp_fn(self, eng, it, session, cost, outp):
        '''
        This method
        eng: 
        it:  
        session: 
        cost:
        outp: 
        '''
        Y_true =[]
        Y_score=[]
        
        cl_list = []
        for b in range(self.bin):
            cl = outp['class_%d'%(b+1)]
            cl_list.append(cl)
        #cl1 = outp['class_1']
        #cl2 = outp['class_2']
        #cl3 = outp['class_3']
        #we get the accuracy of our rules on the test data
        for i in range(self.L):
            Y_true.append ( self.DataSets[self.TEST_SET_INDEX][1][i])
            #Y_score.append( np.argmax( [cl1[i][0] ,cl2[i][0], cl3[i][0] ]))
            Y_score.append(np.argmax( [cl[i][0] for cl in cl_list] ))
        
        acc = accuracy_score(Y_true, Y_score)
        print('***********************************')
        print('accuracy score = ',  acc)
        print('***********************************')

        return acc


    def rundNLunit(self, BS):

        parser = argparse.ArgumentParser()

        parser.add_argument('--CHECK_CONVERGENCE',default=0,help='Check for convergence',type=int)
        parser.add_argument('--SHOW_PRED_DETAILS',default=0,help='Print predicates definition details',type=int)

        parser.add_argument('--PRINTPRED',default=1,help='Print predicates',type=int)
        parser.add_argument('--SYNC',default=0,help='Use L2 instead of cross entropy',type=int)
        parser.add_argument('--L2LOSS',default=1,help='Use L2 instead of cross entropy',type=int)
        parser.add_argument('--BS',default=BS,help='Batch Size',type=int)
        parser.add_argument('--T',default=1,help='Number of forward chain',type=int)
        parser.add_argument('--LR_SC', default={ (-1000,2):.005 ,  (2,1e5):.0051} , help='Learning rate schedule',type=dict)
        parser.add_argument('--ITEM_REMOVE_ITER',default=10000 ,help='length period of each item removal',type=int)
        parser.add_argument('--MAXTERMS',default=3 ,help='Maximum number of terms in each clause',type=int)
        parser.add_argument('--L1',default=.01 ,help='Penalty for maxterm',type=float)
        parser.add_argument('--L2',default=.01 ,help='Penalty for distance from binary',type=float)
        parser.add_argument('--L3',default=.000 ,help='Penalty for distance from binary',type=float)
        parser.add_argument('--L4',default=1.0 ,help='Penalty for distance from binary',type=float)
        parser.add_argument('--ALLTIMESTAMP',default=0 ,help='Add loss for each timestamp',type=int)
        parser.add_argument('--LR', default=.003 , help='Base learning rate',type=float)
        parser.add_argument('--FILT_TH_MEAN', default=.2 , help='Fast convergence total loss threshold MEAN',type=float)
        parser.add_argument('--FILT_TH_MAX', default=.2 , help='Fast convergence total loss threshold MAX',type=float)
        parser.add_argument('--OPT_TH', default=.05 , help='Per value accuracy threshold',type=float)
        parser.add_argument('--PLOGENT', default=.50 , help='Crossentropy coefficient',type=float)
        parser.add_argument('--BETA1', default=.90 , help='ADAM Beta1',type=float)
        parser.add_argument('--BETA2', default=.999 , help='ADAM Beta2',type=float)
        parser.add_argument('--EPS', default=1e-6, help='ADAM Epsillon',type=float)
        parser.add_argument('--GPU', default=1, help='Use GPU',type=int)
        parser.add_argument('--ITER', default=self.iterations, help='Maximum number of iteration',type=int) #100*20
        parser.add_argument('--ITER2', default=100, help='Epoch',type=int)
        parser.add_argument('--LOGDIR', default='./logs/Logic', help='Log Dir',type=str)
        parser.add_argument('--TB', default=0, help='Use Tensorboard',type=int)
        parser.add_argument('--ADDGRAPH', default=1, help='Add graph to Tensorboard',type=int)
        parser.add_argument('--CLIP_NORM', default=0, help='Clip gradient',type=float)

        parser.add_argument('--PRINT_WEIGHTS',default=0,help='Print raw weights',type=int)
        parser.add_argument('--SEED',default=0,help='Random seed',type=int)
        parser.add_argument('--BINARAIZE', default=0 , help='Enable binrizing at fast convergence',type=int)
        parser.add_argument('--MAX_DISP_ITEMS', default=50 , help='Max number  of facts to display',type=int)
        parser.add_argument('--W_DISP_TH', default=.1 , help='Display Threshold for weights',type=int)
        parser.add_argument('--DISP_BATCH_VALUES',default=[],help='Batch Size',type=list)

        parser.add_argument('--CONT_INTERVALS', default=self.num_cont_intervals, help='Number of continuous ranges', type=int)
        parser.add_argument('--CONT_VARIABLES', default=self.names, help='Number variables', type=list)

        #parser.add_argument('--ATOM_NAMES',default=self.pred_trans_name_list, help='Dict of predicate names', type=dict)



        args = parser.parse_args()

        print('displaying config setting...')
        for arg in vars(args):
            print( '{}-{}'.format(arg, getattr(args, arg)))

        self.model = ILPRLEngine( args = args, predColl = self.predColl, bgs = self.bgs, disp_fn = self.disp_fn)
        highest_acc = self.model.train_model()
        #ruleDefList = self.model.ruleDefintionStringList
        return highest_acc

    def retrainModel(self):
        '''
        clear out NN params for model
        '''
        self.model.retrain_model()
