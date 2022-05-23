#from _typeshed import Self
#rom typing import OrderedDict
import numpy as np
import random
import collections
from time import sleep
from datetime import datetime
import sys
import tensorflow.compat.v1 as tf
import os.path
from .mylibw import *
from .dataParse import *
from itertools import product
from itertools import combinations_with_replacement
from .PredicateLibV5 import *
tf.compat.v1.disable_eager_execution()

class ILPRLEngine(object):
    def __init__(self, args, predColl, bgs, disp_fn=None):
        print( 'Tensorflow Version : ', tf.__version__)
        #tf.set_random_seed(self.args.SEED)
        self.args=args
        self.predColl=predColl
        self.bgs=bgs
        self.disp_fn=disp_fn
        config = tf.ConfigProto( device_count = {'GPU': self.args.GPU})
        config.allow_soft_placement =True
        config.log_device_placement = True
        self.session = tf.Session(config = config)
        self.plogent = tf.placeholder("float32", [], name='plogent')
        self.index_ins = OrderedDict({})
        self.X0 = OrderedDict({})
        self.target_mask = OrderedDict({})
        self.target_data = OrderedDict({})
        self.target_difference = OrderedDict({}) #store difference between body of class predicates here
        #instantiate tensors for our corresponding predicates
        for p in self.predColl.preds:
            #tensor constant for each class
            self.index_ins[p.name] = tf.constant(self.predColl.InputIndices[p.name]) #for class1, array([]), shape (1,1,0) 
            #tensor for each class 
            self.X0[p.name] = tf.placeholder("float32", [self.args.BS, self.predColl[p.name].pairs_len], name='input_x_' + p.name) #(40,1)
            if p.pFunc is None:
                continue
            #for each class, tensor of size 40~based on the background fact size
            self.target_data[p.name] = tf.placeholder("float32", [self.args.BS, self.predColl[p.name].pairs_len] , name='target_data_' + p.name) #(40,1)
            self.target_mask[p.name] = tf.placeholder("float32", [self.args.BS, self.predColl[p.name].pairs_len] , name='target_mask_' + p.name) #(40,1)

            #tensors for the weight differences, attempt to code below
            self.target_difference[p.name] = tf.placeholder("float32", [self.args.BS, self.predColl[p.name].pairs_len] , name='target_mask_' + p.name) #(40,1)

        self.thresholds_lt = {}
        self.thresholds_gt = {}
        self.continuous_inputs = {}
        #IMPORTANT; here we learn the less than/greater than values for the particular continuous predicates
        #this code will be relevant when we work on the RRL tasks
        for v in self.predColl.cnts:
            self.thresholds_gt[v.name] = tf.get_variable( 'th_gt_' + v.name , shape=(v.no_gt), 
                initializer=tf.constant_initializer(v.gt_init, dtype=tf.float32) ,collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'CONTINUOUS'])
            self.thresholds_lt[v.name] = tf.get_variable( 'th_lt_' + v.name , shape=(v.no_lt), 
                initializer=tf.constant_initializer(v.lt_init, dtype=tf.float32) ,collections=[ tf.GraphKeys.GLOBAL_VARIABLES,'CONTINUOUS'] )
            #arg BS (Batch Size) determines the input size, key variable (6)
            self.continuous_inputs[v.name] = tf.placeholder("float32", [self.args.BS,v.dim,self.args.T] ,name='continous_input_'+v.name)

        self.define_model()
        print("summary all variables")

        for k in tf.trainable_variables():
            if( isinstance(k, tf.Variable) and len(k.get_shape().as_list())>1):
                print(str(k))
                if (self.args.TB == 1):
                    tf.histogram(k.name, k)
                    if len(k.get_shape().as_list())==2:
                        tf.summary.image(k.name, tf.expand_dims( tf.expand_dims(k,axis=0),axis=3))
                    if len(k.get_shape().as_list())==3:
                        tf.summary.image(k.name, tf.expand_dims( k,axis=3))
        if self.args.TB==1:
            self.all_summaries = tf.summary.merge_all()
    ############################################################################################
    def check_weights(self, sess, w_filt):
        for p in self.predColl.preds:
            wts = tf.get_collection( p.name)
            if len(wts)==0:
                continue
            wvs = sess.run( wts )

            for wt,wv in zip(wts,wvs):
                if not wt.name.endswith(w_filt) :
                    continue
                wv_sig = p.pFunc.conv_weight_np(wv)

                sumneg = np.sum( np.logical_and(wv_sig>.1,wv_sig<.9))
                if sumneg > 0:
                    print( "weights in %s are not converged yet :  %f"%(wt.name,sumneg))
                    return False
        return True
    ############################################################################################
    def filter_predicates( self,sess , w_filt,th=.5  ):
        old_cost,_=self.runTSteps(sess)
        for p in self.predColl.preds:
            wts = tf.get_collection( p.name)
            if len(wts)==0:
                continue
            wvs = sess.run( wts )

            for wt,wv in zip(wts,wvs):
                if not wt.name.endswith(w_filt) :
                    continue


                wv_sig = p.pFunc.conv_weight_np(wv)
                for ind,val in  np.ndenumerate(wv_sig):

                    if val>.5:
                        wv_backup = wv*1.0
                        wv[ind]=-20
                        sess.run( wt.assign(wv))
                        cost,_=self.runTSteps(sess)

                        if cost-old_cost >th :
                            wv = wv_backup*1.0
                        else:
                            old_cost=cost
                            print( 'removing',wt,ind)

                sess.run( wt.assign(wv))   
    ############################################################################################
    def filter_predicates2( self,sess ,  th=.5  ): 
        pass
    ############################################################################################
    def get_sensitivity_factor( self,sess , p ,target_pred ):
        pass
    ############################################################################################
    def get_sensitivity_factor1( self,sess , p ,target_pred ):    
        pass

    ############################################################################################
    def binarize( self,sess   ):
        for p in self.predColl.preds:
            wts = tf.get_collection( p.name)
            if len(wts)==0:
                continue
            wvs = sess.run( wts )
            for wt,wv in zip(wts,wvs):
                wv = wv*1.6
                s = 20
                wv [ wv>s] =s
                wv[wv<-s] = -s
                sess.run( wt.assign(wv))

    ############################################################################################
    def define_model(self):
        XOs = OrderedDict( self.X0 )#set of target preds, class1, etc.
        L3=0
        self.XOTS=OrderedDict()

        # forward  chaining for dNL-ILP
        # t steps in forward chaining
        for t in range(self.args.T): #Tmax is only 1 here...

            olditem=OrderedDict()
            #for each  input tensor for the target preds
            for i in XOs:
                olditem[i] = tf.identity( XOs[i])#Return a Tensor with the same shape and contents as input.
            #for each predicate in the set of target predicates P
            for p in self.predColl.preds:

                if p.name=="CNT":
                    lenp = len(p.pairs)
                    px = np.zeros( (self.args.BS,lenp) , np.float)
                    if t<lenp:
                        px[:,t]=1
                    else:
                        px[:,-1]=1

                    XOs[p.name] = tf.constant(px,tf.float32)
                    continue

                if p.pFunc is None:
                    continue

                #check to see that we have continuous intensional? predicates
                if len( self.predColl.cnts) >0 :
                    inp_continous=[]
                    #iterate over each  ground atom for predicate p~class_k: e ∈ G_p
                    for v in self.predColl.cnts:
                        if v.dim>1:
                            continue
                        '''
                        The following code demonstrates the continous fuzzy predicate function. 
                        each continuous variable x we define k lower-boundary predicates gt x i (x, l x i ) 
                        as well as k upper-boundary predicates lt x i (x, u x i ) where i ∈ {1, . . . , k}. 
                        We let the boundary values l x i ’s and u x i ’s be trainable weights and
                        we define the upper-boundary and lower-boundary predicate functions as:
                                F gt x i = σ(c (x − u x i )) , F lt xi = σ(−c (x − l x i ))
                        The thresholds_gt, thresholds_lt correspond to the boundary values, which are also 
                        trainable weights.
                        '''
                        #  x1(6,2) =  TensorInput X1 (6,1) - upperbound tensor vals  (1,2)
                        x1 = self.continuous_inputs[v.name][:,:,t] - tf.expand_dims( self.thresholds_gt[v.name] , 0) #TensorShape[6,2]
                        x2 = self.continuous_inputs[v.name][:,:,t] - tf.expand_dims( self.thresholds_lt[v.name] , 0)
                        x1 = sharp_sigmoid( x1 , 20)
                        x2 = sharp_sigmoid( -x2 , 20)

                        cond1 = p.exc_cnt is not None and v.name in p.exc_cnt
                        cond2 = p.inc_cnt is not None and v.name not in p.inc_cnt

                        if cond1 or cond2:
                            pass
                            # inp_continous.append(x1*0)
                            # inp_continous.append(x2*0)
                        else:
                            #new)value = new_value V F p i | θ
                            inp_continous.append(x1) #list of fuzzy bools for cnt preds, F gt x i = σ(c (x − u x i ))
                            inp_continous.append(x2) #[X1,X2,SinX1,SinX2, etc..]

                    if len(inp_continous)==0:
                        len_continous = 0
                    else:

                        inp_continous = tf.concat(inp_continous,-1) #Tensor (6,24)
                        len_continous = inp_continous.shape.as_list()[-1]
                else:
                    len_continous = 0

                if self.args.SYNC==1:
                    x = tf.concat( list( olditem.values() ), -1)
                else:
                    #get the input tensors for the target preds
                    x = tf.concat( list( XOs.values() ), -1) #Tensor (6,3)
                #gather (params = x Tensor (6,3), indices = (class1,..classk)) for class 1 Tensor (1,1,0)
                #basically we are getting our 
                xi=tf.gather( x  ,self.index_ins[p.name],axis=1  ) #Tensor (6,1,1,0)
                s = xi.shape.as_list()[1]*xi.shape.as_list()[2] #1




                self.xi = xi
                if p.Lx>0:
                    xi = tf.reshape( xi, [-1,p.Lx])
                    if p.use_neg:
                        xi = tf.concat( (xi,1.0-xi) ,-1)
                #always true for us as we only use cnt preds
                if len( self.predColl.cnts) >0  and p.use_cnt_vars: 
                    #this code usually creates multiple 'copies' of preds for our fuzzy bool,
                    #however as we are using cnts with no discret variable input (A,B,C), but using
                    #real values, we don't need ot create copies
                    cnt_s = tf.tile( inp_continous,(s,1) ) #Tensor [6,24]
                    if p.Lx>0: #we combine the input to our class tensor xi with our cnt predicates upper/lower bounds
                        xi = tf.concat(  (xi,cnt_s),-1)
                    else:
                        xi = cnt_s #Tensor (6,24)


                l = xi.shape.as_list()[0] #6, number of cnt preds
                # if t==0:
                #     print( 'input size for F (%s) = %d'%(p.name,l))



                with tf.variable_scope( "ILP", reuse=tf.AUTO_REUSE, auxiliary_name_scope=False):
                    xi = p.pFunc.pred_func(xi,self.continuous_inputs,t) #Tensor (6,1)
                    if type( xi ) in  (tuple,list):
                        for a in xi[1]:
                            self.continuous_inputs[a] = copy.deepcopy( xi[1][a] )
                        xi = xi[0]




                xi = tf.reshape( xi  , [self.args.BS,]+self.index_ins[p.name].shape.as_list()[:2] ) #Tensor (6,1,1)
                #Boolean function F_c (x i , m i ), with and_op acting as memberhsip weights#
                xi =  1.0-and_op( 1.0-xi,-1) #Tensor (6,1) x V y disjunction function

                #fuzzy boolean functions
                '''
                Neural Conjunction and Disjunction Layers
                Let {x 1 , . . . , x n } be the input vector consists of n Boolean variables and consider the case
                of conjunction function. To each variable x i we associate a Boolean flag m i such that it
                control the inclusion or exclusion of the term x i in the resulting conjunction function. To
                this end, we define the following Boolean function:
                f conj = (m 1 ∨ x 1 ) ∧ · · · ∧ (m n ∨ x n )

                n order to design a differentiable network, we use the continuous relaxation of Boolean,
                i.e., we assume fuzzy Boolean values are real values in the range [0, 1] and we use 1 (True)
                and 0 (False) representations for the two states of a binary variable. We also define the fuzzy
                unary and dual Boolean functions of two Boolean variables x and y as:
                x̄ = 1 − x
                x ∧ y = xy
                ȳ = 1 − y
                x ∨ y =1 − (1 − x)(1 − y)
                '''
                #NOT USED!
                L3+=  tf.reduce_max( xi*(1.0-xi)) #Interpretability Loss L int = E where, E m(1 − m)
                #It could mean our L4 error could be included here, keeping class defs consistent 

                if p.Fam =='and':
                    XOs[p.name] = XOs[p.name] *  tf.identity(xi) #X ^ Y
                if p.Fam =='eq':
                    XOs[p.name] = tf.identity(xi) #just says the Tensor for our target class is equivalent to out cnt preds lower/upper
                if p.Fam =='or':
                    XOs[p.name] = 1.0 - (1.0-XOs[p.name] )*  (1.0-tf.identity(xi) ) # x V y


            self.XOTS[t] = dict()
            self.XOTS[t].update(XOs)
            #Enf of forward chaining dNL bool module

        '''
        Def of the the Loss function
        '''
        L1=0
        L2=0
        #L4 will be for our costum loss function designed to keep the class rule defs consistent
        L4=0

        for p in  self.predColl.preds:
            vs = tf.get_collection( p.name)
            for wi in vs:
                if '_AND' in wi.name: #isolate only conj. layer
                    wi = p.pFunc.conv_weight(wi)

                    L2 += tf.reduce_mean( wi*(1.0-wi))#Interpretability Loss L int = E where, E m(1 − m)

                    s = tf.reduce_sum( wi,-1)
                    L1 += tf.reduce_mean(  tf.nn.relu( s-self.args.MAXTERMS)  )

        #LOSS FUNCTION, we need 
        self.XOs=XOs
        self.loss_gr = tf.constant(0.,tf.float32)
        self.loss = tf.constant(0.,tf.float32)

        for p in self.predColl.preds:
            if p.pFunc is None:
                continue

            if self.args.L2LOSS==1:
                err = ( self.target_data[p.name] - XOs[p.name] ) * self.target_mask[p.name]
                err = tf.square(err)
                self.loss_gr +=   tf.reduce_mean(err,-1)
            else:
                err = neg_ent_loss (self.target_data[p.name] , XOs[p.name] ) * self.target_mask[p.name]
                self.loss_gr +=  tf.reduce_mean(err,-1)

            print("########################-----------LOSS-------------#############")
            print(p.name)
            print(self.target_data[p.name])
            print(XOs[p.name])
            print(self.target_mask[p.name])
            print('####################################################################%')
            
            loss =  neg_ent_loss (self.target_data[p.name] , XOs[p.name] ) * self.target_mask[p.name]
            self.loss += tf.reduce_sum ( loss )
        

        '''
        Start of the code for our L4 loss function. The idea here is to get the weights from conj. layer
        for each predicate, much like for L2 and L1. use the weights to determine whether a given
        cnt pred is in the body of our class defintion. As there are a potential of 8 lower/upperbound versions
        of a single predicate as we have two conj layers (2x24) so we index by 4 where the first two positions
        are SinX1 > a SinX1> b then lower bound SinX1 <
        '''
        num_preds =len(self.predColl.cnts)
        sum_pred_wts = tf.zeros(num_preds) #we add the total count for a pred name here
        num_pred_same = self.args.CONT_INTERVALS * 2 #this is total number of cnt preds for a specific transformation (sinX1, etc.)
        
        for p in self.predColl.preds: #iterate over each class
            vs = tf.get_collection(p.name)
            for wi in vs:
                if '_AND' in wi.name: #isolate only conj. layer
                    wi = p.pFunc.conv_weight(wi)
                    pred_max=[]
                    idx_start=0
                    for cnt in range(num_preds): #iterate through number of potential body preds
                        cnt+=1
                        pred_max.append(tf.round(tf.reduce_max(wi[:,idx_start:num_pred_same*cnt])))
                        idx_start = num_pred_same*cnt
                        #np_pred_max = np.array(pred_max)
            sum_pred_wts = sum_pred_wts + pred_max
        #islote values of K, and 0, K indicates that predicate 
        #exists in the bodies of all preds, and 0 means that pred 
        #is not present in the body of any of the K classes
        zero_vector = tf.zeros(shape=(num_preds,))
        bool_zero_mask = tf.not_equal(sum_pred_wts, zero_vector)
        omit_zeros = tf.boolean_mask(sum_pred_wts, bool_zero_mask)
        size_zero = tf.size(omit_zeros)#.shape[0]
        #now we remove the cases where all K classes are present
        three_vector =tf.fill((size_zero,),3.0)
        bool3_mask = tf.not_equal(omit_zeros, three_vector)
        omit_threes = tf.boolean_mask(omit_zeros, bool3_mask)
        #get the lenght of the final tensor as its number of preds 
        #that are inconsistent with the class defs
       
        L4 =tf.cast( tf.size(omit_threes, out_type=tf.dtypes.int64),tf.dtypes.float32)




    

        #try to get the weights for the predicates (X1, sinX2, prodX1X2) and calc the difference
        #ISSUE: I am getting the weights for two defintions for a class since there are two defintions in with
        #       sin(X1) + exp(X2). but future cases may have only one or possibly more. will need to make this 
        #       modular and collec
        #ISSUE: Now the predicate is including a lot of extra predicates, and in sme cases incorrect predicates, 
        #       so the heuristic needs to be changed a bit. It could the interval is including other predicates 
        #       or by focusing on just 1 our the interval list we are optimizing for something unseen, this might
        #       explain why there are multiple of the same predicate in some of the bodies for the class defs.
        #ISSUE: As predicates need to pass a threshold to be included in the body (0.2) it might be the case 
        #       that we are optimizing for predicates in the body that should not be present. this check is not made
        #       by p.pFunc.getTensorWeights()
        total_weight_difference = None
        temp =None
        for p in self.predColl.preds:
            if p.pFunc is not None:
                temp = p.pFunc.getTensorWeights(self.args.CONT_INTERVALS)
                cnt_vars = self.predColl.get_continuous_var_names_novar(p)
                print(cnt_vars) # to get a visualizaion of the predicate names, and the  (80,)
                defintions_count = len(temp) #may be used later for all defintions
                for d in range(1): #(defintions_count), but lets just try the first def for now
                    pred_body_weights = temp[d]
                    if total_weight_difference is None:
                        total_weight_difference = pred_body_weights
                    else: #ISSUE: the defintion for 1-80 includes predicates associated with specific ranges X1<a, a<X1<b, etc.
                        #       we need to group the pedicates to just the name, not the range, and see if they are present in the body
                        #       we are not worried about the ranges, just that the predicate names are consistent for the bodies of each class
                        # the current strategy is to find the largest value for the range of predicates, indicating the presence
                        # of that predicate in the body of the class, then take the difference between that and the other 
                        # class' weight for the corresponding predicate. The assumption is that for two classes which should have
                        # body definitions comprising the same predicates, the tensor weights will be approximate in magnitude
                        # so we are taking the sum of the absolute difference between these weights and trying to minimize that difference during
                        # learning
                        total_weight_difference = tf.math.abs(tf.math.subtract(total_weight_difference, pred_body_weights))
        #print(total_weight_difference)  #the sum of the absolute difference between the body weights for each class predicate
        #sum_abs_diff_body_weights = sum(total_weight_difference)
        
        #self.loss_gr += 0.01*tf.reduce_sum(total_weight_difference)


        #here we will try to get the weight intervals, to ensure that only one is true by optimizing the weights
        for p in self.predColl.preds:
            cnt_vars = self.predColl.get_continuous_var_names_novar(p)
            unique_pred_names = set(cnt_vars)
            for body_pred in unique_pred_names:
                print(body_pred)
                tempAND, tempOR = p.pFunc.getTensorIntervalWeights( body_pred, cnt_vars ,self.args.CONT_INTERVALS)
                #self.loss_gr += 0.001 * tf.nn.l2_loss(tempAND)
                #self.loss_gr += 0.001 * tf.nn.l2_loss(tempOR)



        self.loss_gr  += ( self.args.L1*L1 + self.args.L2*L2+self.args.L3*L3 + self.args.L4*L4 )
        # can we output current list of preds for a class pred
        #how would we use this to add a heuristic to limit 
        #difference between body of class preds
        #for p in self.predColl.preds:
        #    if p.pFunc is None:
        #        continue
        #    min_entropy_loss(self.target_data[p.name], XOs[p.name], self.predColl, p.name)

        #print(fd)
        self.lastlog=10
        self.cnt=0
        self.counter=0
        self.SARG=None




    ############################################################################################
    # execute t step forward chain
    def runTSteps(self,session,is_train=False,it=-1):
        # if self.SARG is None:
        #SARG will contain all background facts based on the data/transformations on data
        self.SARG = dict({})
        bgs = self.bgs(it,is_train) #contains background values from data
        self.SARG[self.plogent] =self.args.PLOGENT
        
        for p in self.predColl.preds:
            self.SARG[self.X0[p.name]] = np.stack( [bg.get_X0(p.name) for bg in bgs] , 0 )
            if p.pFunc is None:
                continue
            self.SARG[self.target_data[p.name]] = np.stack( [ bg.get_target_data(p.name) for bg in bgs] , 0 )
            self.SARG[self.target_mask[p.name]] = np.stack( [ bg.get_target_mask(p.name) for bg in bgs] , 0 )

        for c in self.predColl.cnts:           
            self.SARG[self.continuous_inputs[c.name]] =   np.stack( [ bg.continuous_vals[c.name] for bg in bgs] , 0 )

        self.SARG[self.LR] = .001
        try:
            if is_train:
                if bool(self.args.LR_SC) :
                    for l,r in self.args.LR_SC:
                        if self.lastlog >= l and self.lastlog < r:
                            self.SARG[self.LR] = self.args.LR_SC[(l,r)]
                            break

        except:
            self.SARG[self.LR] = .001
        # here we train the model to get ouput
        #train_op has the tensors for the relevant threshold values
        if is_train:
            _,cost,outp =  session.run( [self.train_op,self.loss,self.XOs ] , self.SARG)
        else:
            cost,outp,xots =  session.run( [self.loss,self.XOs ,self.XOTS ] , self.SARG )

        try:
            self.lastlog = cost
        except:
            pass
        return cost,outp

    ############################################################################################
    def train_model(self):
        session  = self.session

        t1 =  datetime.now()
        print ('building optimizer...')
        self.LR = tf.placeholder("float", shape=(),name='learningRate')

        #Loss here as as well
        loss = tf.reduce_mean(self.loss_gr)
       
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.LR, beta1=self.args.BETA1,
                       beta2=self.args.BETA2, epsilon=self.args.EPS,
                       use_locking=False, name='Adam')

        self.train_op  = self.optimizer.minimize(loss)



        t2=  datetime.now()
        print ('building optimizer finished. elapsed:' ,str(t2-t1))
        #we get the various tensors for which we will optimize
        #included are the threshold values for the continuous predicates
        init = tf.global_variables_initializer()
        var = session.run(init)

        if self.args.TB==1:
            train_writer = tf.summary.FileWriter(self.args.LOGDIR, session.graph)
            train_writer.close()

        print( '***********************')
        print( 'number of trainable parameters : {}'.format(count_number_trainable_params()))
        print( '***********************')


        start_time = datetime.now()

        highest_acc = 0

        # for t in {1,...,tmax} do
        for i in range(self.args.ITER):

            #forward chaining of the dNL neuron    
            cost,outp= self.runTSteps(session,True,i)


            if i % self.args.ITER2 == 0 and not np.isnan(np.mean(cost)):
                

                cost,outp= self.runTSteps(session,False,i)

                if self.disp_fn is not None:
                    acc = self.disp_fn(self, i//self.args.ITER2,session,cost,outp)
                    if acc >= highest_acc:
                        highest_acc = acc

                now = datetime.now()
                print('------------------------------------------------------------------')
                errs=OrderedDict({})
                
                #for p \in P {class1, class2}
                for p in self.predColl.preds:
                    if p.pFunc is None:
                        continue
                    if np.sum(self.SARG[self.target_mask[p.name]]) >0:
                        errs[p.name] = np.sum (  (np.abs(outp[p.name]-self.SARG[self.target_data[p.name]] )) * self.SARG[self.target_mask[p.name]] )

                print( 'epoch=' ,i//self.args.ITER2 , 'cost=', np.mean(cost),  'elapsed : ',  str(now-start_time)   ,'mismatch counts', errs)
                names=[]

                #displaying outputs ( value vectors)
                for bs in self.args.DISP_BATCH_VALUES:
                    if bs>0:
                        break
                    cnt = 0

                    for p in self.predColl.preds:
                        print_names=[]
                        if p.pFunc is None:
                            continue

                        mask = self.SARG[self.target_mask[p.name]]
                        target = self.SARG[self.target_data[p.name]]
                        if np.sum(mask) >0:

                            for ii in range( p.pairs_len ):
                                if mask[bs,ii]==1:
                                    if cnt<self.args.MAX_DISP_ITEMS:
                                        print_names.append(  '[('+ ','.join( p.pairs[ii]) +')],[%2.01f,%d]  '%(outp[p.name][ bs,ii],target[bs,ii]))
                                        if  abs(outp[p.name][bs,ii]-target[bs,ii]) >.3:
                                            print_names[-1] = '*'+print_names[-1]
                                        else:
                                            print_names[-1] = ' '+print_names[-1]


                                        if  cnt%10==0:
                                            print_names[-1] = '\n' +print_names[-1]
                                        cnt+=1

                                    else:
                                        break
                        print( ' , '.join(print_names) )




                # remove unncessary terms if near optimzed solution is achieved or preprogrammed to do so
                err = [  (np.abs(outp[p.name]-self.SARG[self.target_data[p.name]] )) * self.SARG[self.target_mask[p.name]]  for p in self.predColl.preds if p.pFunc is not None]
                errmax = np.max (  [ e.max() for e in err])
                try:

                    if i>0 and ( (i//self.args.ITER2)%self.args.ITEM_REMOVE_ITER==0) :
                        print ( 'start removing non necessary clauses')
                        self.filter_predicates(session,'OR:0')
                        self.filter_predicates(session,'AND:0')
                except:
                    pass
                if  np.mean(cost)<self.args.FILT_TH_MEAN  and errmax<self.args.FILT_TH_MAX or ( np.mean(cost)<self.args.FILT_TH_MEAN and i%1000==0 ):

                    should_remove=True
                    for ii in range(20):
                        cost,outp= self.runTSteps(session,False)
                        err = [  (np.abs(outp[p.name]-self.SARG[self.target_data[p.name]] )) * self.SARG[self.target_mask[p.name]]  for p in self.predColl.preds if p.pFunc is not None]
                        errmax = np.max (  [ e.max() for e in err])


                        if  np.mean(cost)<self.args.FILT_TH_MEAN  and errmax <self.args.FILT_TH_MAX or ( np.mean(cost)<self.args.FILT_TH_MEAN and i%1000==0 ):
                            pass
                        else:
                            should_remove = False
                            break
                    should_remove = should_remove
                    if should_remove:
                        print ( 'start removing non necessary clauses')

                        self.filter_predicates(session,'OR:0')
                        self.filter_predicates(session,'AND:0')
                        if self.args.BINARAIZE==1:
                            self.binarize(session)

                            self.filter_predicates(session,'OR')
                            self.filter_predicates(session,'AND')
                            cost,outp= self.runTSteps(session,False)

                        if self.args.CHECK_CONVERGENCE==1:
                            self.check_weights(session,'AND:0')
                            self.check_weights(session,'OR:0')
                print(self.args.PRINTPRED)
                
                self.args.PRINTPRED  = 1
                #display learned predicates in string format
                for p in self.predColl.preds:
                    print(p.name)
                    print(p.inp_list)
                    cnt_vars = self.predColl.get_continuous_var_names_novar(p)
                    if p.pFunc is not None:
                        #def get_func(self,session,names,threshold=.2,print_th=True):
                        s = p.pFunc.getSimpleFunc( session,cnt_vars,threshold=self.args.W_DISP_TH)
                        print("-----------------------------------------------------------------------")
     

                    
                #print the class1, class2, ... body defintions with membership weights
                if self.args.PRINTPRED :
                    try:
                        gt,lt = session.run( [self.thresholds_gt,self.thresholds_lt] )
                        #print(gt)
                        #print(lt)
                        print("-----------------------------------------------------------------------")
                        for p in self.predColl.preds:

                            cnt_names = self.predColl.get_continuous_var_names(p,gt,lt)

                            if p.pFunc is None:
                                continue
                            if p.use_cnt_vars:
                                inp_list = p.inp_list  + cnt_names
                            else:
                                inp_list = p.inp_list

                            if p.pFunc is not None:
                                s = p.pFunc.get_func( session,inp_list,threshold=self.args.W_DISP_TH)
                                if s is not None:
                                    if len(s)>0:
                                        print( p.name+ '(' + ','.join(variable_list[0:p.arity]) + ')  \n'+s)

                    except:
                        print('there was an exception in print pred')
                # display raw membership weights for predicates
                if self.args.PRINT_WEIGHTS==1:
                    wts = tf.trainable_variables( )
                    wvs = session.run( wts )
                    for t,w in zip( wts,wvs):
                        if '_SM' in t.name:
                            print( t.name, np.squeeze( w.argmax(-1) ) )
                        else:
                            print( t.name, myC( p.pFunc.conv_weight_np(w) ,2) )

                # check for optimization
                err = [  (np.abs(outp[p.name]-self.SARG[self.target_data[p.name]] )) * self.SARG[self.target_mask[p.name]]  for p in self.predColl.preds if p.pFunc is not None ]
                errmax = np.max (  [ e.max() for e in err])

                if np.mean(cost)<self.args.OPT_TH  and ( np.mean(cost)<.0 or  errmax<.09 ):


                    if self.args.CHECK_CONVERGENCE==1:
                        if self.check_weights(session,'OR:0')  and self.check_weights(session,'AND:0')  :

                            print('optimization finished !')

                            return
                    else:
                        print('optimization finished !')

                        return

                start_time=now
        print(self.X0)
        print(self.predColl)
        for p in self.predColl.preds:
            #print(self.get_sensitivity_factor( self.session,'class_1', p))
            print(p)
        for p in self.predColl.preds:
            #print(self.get_sensitivity_factor( self.session,'class_1', p))
            print(p.name)

        print(self.filter_predicates)
        print("Highest accuracy is ",highest_acc)


        #The following code will be used to get the current rules and then pass them to the parseTransformation 
        #for use in the second phase of the non-linear learning.
        #TODO MAKE THIS DYNAMIC, NOT HAND CRAFTED
        #pred_parse_list = {'X1' : 0, 'X2': 0, 'X1Exp': 0, 'X2Exp': 0, 'X1Square':0 , 'X2Square':0, 'X1Sine':0 , 'X2Sine':0 }
        pred_parse_list = {v: 0 for v in self.args.ATOM_NAMES}

        if self.args.PRINTPRED :
            for p in self.predColl.preds:
                print(p.name)
                #print(p.inp_list)
                cnt_vars = self.predColl.get_continuous_var_names_novar(p)
                if p.pFunc is not None:
                    #def get_func(self,session,names,threshold=.2,print_th=True):
                    s = p.pFunc.getSimpleFunc( session,cnt_vars,threshold=self.args.W_DISP_TH)
                    print("-------------------parse-Transform--------------------------------------")

        predlist = parseTransformation(session, self.predColl, self.args, pred_parse_list,self.args.CONT_VARIABLES)
        return predlist
  





