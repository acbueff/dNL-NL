import tensorflow.compat.v1 as tf
import numpy as np
import copy
import inspect

def OR(x,y):
    pass
#####################################################
def XOR(x,y):
    pass
#####################################################
def NOT(x):
    pass
#####################################################
def mysplit(x,sz,axis=-1):
    pass
#####################################################
def update_dic( d , i , v, mode='max'):
    pass
#####################################################
def read_by_tokens(fileobj):
    pass
#####################################################
def prinT(var):
    pass
#####################################################
def partition_range( total_size, partition_size):
    pass
#####################################################
def add_neg(x):
    pass
#####################################################
def myC(x,n=2):
    k=10**2
    return np.round(x*k)/k
#####################################################
def clip_grads_val( optimizer,loss,min_val,max_val,global_state=None):
    pass
#####################################################
def custom_grad(fx,gx):
    pass
#####################################################
def FC(inputs,sizes,activations=None,name='fc'):
    pass
def weight_variable(shape, stddev=0.01, name='weight'):
    pass
#####################################################
def bias_variable(shape, value=0.0, name='bias'):
    pass
#####################################################
def sig(x,p=1.):
    pass
#####################################################
def npsig(x,p=1.):
    pass
#####################################################
BSP=50
sss="-----------------------------------------------\n"
and_op = lambda x,ax=-1:   tf.reduce_prod( x  , axis=ax,name='my_prod') 
or_op = lambda x,ax=-1:  1.0-tf.reduce_prod( 1.0-x  , axis=ax) 
####################################################
def make_batch(batch_size,v):
    pass
#####################################################
def relu1(x):
    return tf.nn.relu ( 1.0 - tf.nn.relu(1.0-x) )
#####################################################
def leaky_relu1(x):
    pass
#####################################################
def neg_ent_loss(label,prob,eps=1.0e-4):
    return - ( label*tf.log(eps+prob) + (1.0-label)*tf.log(eps+1.0-prob))
#####################################################
'''
can we create a loss function which reduces 
'''
def min_entropy_loss(lable, prob, predColl, pname):
    #cnt_names = predColl.get_continuous_var_names(p,gt,lt)
    for p in predColl.preds:
        print(sss)
        print(p.name)
        cnt_vars = predColl.get_continuous_var_names_novar(p)
        print(p.inp_list)
        print(sss)
        print(cnt_vars)
        print(sss)

    #take the print code and focus on just the variables for a given classificaton
    '''  
      if self.args.PRINTPRED :
        try:
            gt,lt = session.run( [self.thresholds_gt,self.thresholds_lt] )
            print(gt)
            print(lt)
            print("THIS FAR#3##############")
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

            '''
#####################################################
def neg_ent_loss_p(label,prob,p=.5,eps=1.0e-4):
    pass
#####################################################
def sharp_sigmoid(x,c=5):
    cx = c*x
    return tf.sigmoid(cx)
#####################################################
def sharp_sigmoid_np(x,c=5):
    '''
    applys the sigmoid function to the weights to get the likelihoods
    '''
    cx = c*x
    cx = np.clip(cx,-30,30) #Given an interval, values outside the interval are clipped to the interval edges.
    
    return 1./ (1+ np.exp( -cx ) )
#####################################################
def _concat(prefix, suffix, static=False):
    pass
#####################################################
def logic_layer_and(inputs, units,scope=None, col=None ,name="W", trainable=True, sig =1.,mean=0.0,std=2.,w_init=None,rescale=False):
    if isinstance(inputs,tuple) or isinstance(inputs,list):
        inputs = tf.concat(inputs,axis=-1)

    V = inputs
    L = V.get_shape().as_list()[-1]
    
    
    if w_init is not None:
        init = tf.constant_initializer(w_init)
    else:
        if std<0: 
            init = RandomBinary(-std,.551)
        else:
            init = tf.truncated_normal_initializer(mean=mean,stddev=std)

    
    if scope is not None:
        with tf.variable_scope( scope, tf.AUTO_REUSE):
            W= tf.get_variable(name, [units, L ],initializer= init,regularizer=None,dtype=tf.float32,  collections=col, trainable= trainable)
    else:
        W= tf.get_variable(name , [units, L ],initializer= init,regularizer=None,dtype=tf.float32,  collections=col, trainable= trainable)
    

    if sig>0:
        W = sharp_sigmoid(W,sig)
    else:
        W = relu1(W)

    Z = tf.expand_dims(W,axis=0) * (1.0-tf.expand_dims(V,axis=1) )
    S=and_op( 1.0-Z  )

    return S
#####################################################
def logic_layer_or(inputs, units,scope=None, col=None ,name="W", trainable=True, sig =1.,mean=0.0,std=2.,w_init=None,rescale=False):
     
    if isinstance(inputs,tuple) or isinstance(inputs,list):
        inputs = tf.concat(inputs,axis=-1)

    V = inputs
    L = V.get_shape().as_list()[1]
    
    
    if w_init is not None:
        init = tf.constant_initializer(w_init)
    else:
        if std<0: 
            init = RandomBinary(-std,.551)
        else:
            init = tf.truncated_normal_initializer(mean=mean,stddev=std)

    
    if scope is not None:
        with tf.variable_scope( scope, tf.AUTO_REUSE):
            W= tf.get_variable(name, [units, L ],initializer= init,regularizer=None,dtype=tf.float32,  collections=col, trainable= trainable)
    else:
        W= tf.get_variable(name , [units, L ],initializer= init,regularizer=None,dtype=tf.float32,  collections=col, trainable= trainable)

    if sig>0:
        W = sharp_sigmoid(W,sig)
    else:
        W = relu1(W)

    Z = tf.expand_dims(W,axis=0) * tf.expand_dims(V,axis=1)
    S = 1.0-and_op( 1.0-Z )
     
    return S
#####################################################
def logic_layer_and_multi(inputs, units,n1=10,n2=2,scope=None, col=None ,name="W",  sig =1.,mean=0.0,std=2. ):
    pass
#####################################################
def get_nb_params_shape(shape):
    pass
#####################################################
def count_number_trainable_params():
    pass
#####################################################
class RandomBinary(object):
   
    def __init__(self, k,s, seed=0, dtype=tf.float32):
        self.k=k
        self.s=s
        self.seed = seed
        self.dtype = tf.as_dtype(dtype)

    def __call__(self, shape, dtype=None, partition_info=None):
        
        if dtype is None:
            dtype = self.dtype
        
        
        if shape[0]==1 and len(shape)>2:
            inc=1
        else:
            inc = 0 
         

        logit = tf.ones([shape[inc], shape[inc+1]]) /shape[inc+1]
        v1 = tf.multinomial(logit, self.k)
        v2 = tf.one_hot( v1,shape[inc+1])
        v3=tf.reduce_sum(v2,axis=-2)
        v3 = tf.reshape( v3, shape)
        v=  5*(relu1(v3)/1.9-.5 )
        return v 

    def get_config(self):
        return {
            "alpha": self.alpha,
            "seed": self.seed,
            "dtype": self.dtype.name
        }
