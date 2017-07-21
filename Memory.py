'''Created and developed by Akhil K'''
'''Find me at blitzkrieg.akhil@gmail.com'''
import numpy as np
import math
import random

def sigmoid(x):
    return 1./(1+np.exp(-x))

def init(n_cells):
    #Initializes weights in memory cells
    #list[0] is the weight of h_t-1, list[1] is the weight of x_t, list[2] is the bias
    f_tw=[]
    i_tw=[]
    del_cw=[]
    o_tw=[]
    for i in range(n_cells):
        f_tw.append([])
        i_tw.append([])
        del_cw.append([])
        o_tw.append([])
        for j in range(3):
            #Weight ranges from 0.0 to 5.0
            f_tw[i].append(random.random()*5)
            i_tw[i].append(random.random()*5)
            del_cw[i].append(random.random()*5)
            o_tw[i].append(random.random()*5)
    return {'f_tw':f_tw,'i_tw':i_tw,'del_cw':del_cw,'o_tw':o_tw}

def forward_propogation(n_cells,inputs,model):
    f_t=0
    i_t=0
    del_c=0
    o_t=0
    h_t=0
    c_t=0
    exp_out=[]
    #Contains all c_t-1 values at each timestamp
    cell_t=[]
    for i in range(n_cells):
        f_t=sigmoid(model['f_tw'][0]*h_t+model['f_tw'][1]*inputs[i]+model['f_tw'][2])
        i_t=sigmoid(model['i_tw'][0]*h_t+model['i_tw'][1]*inputs[i]+model['i_tw'][2]) 
        del_c=np.tanh(model['del_cw'][0]*h_t+model['del_cw'][1]*inputs[i]+model['del_cw'][2])
        cell_t.append(h_t)
        c_t=f_t*c_t+i_t*del_c
        o_t=sigmoid(model['o_tw'][0]*h_t+model['o_tw'][1]*inputs[i]+model['o_tw'][2])
        h_t=o_t*tanh(c_t)
        exp_out.append(h_t)
    model['exp_out']=exp_out
    model['cell_t']=cell_t
    return model  

def back_propogation(n_cells,inputs,outputs,model):
    return model
def iterator(iterations):
    #Training phase
    inputs=[]
    outputs=[]
    n_cells=int(input("Enter the number of LSTM memory cells required "))
    model=init(n_cells)
    for i in range(iterations):
        inputs.append([])
        outputs.append([])
        for j in range(n_cells):
            inputs[i].append(int(input("Enter an input or 0 for no input")))
        for j in range(n_cells):
            output[i].append(int(input("Enter an output or 0 for no output")))
        model=forward_propogation(n_cells,inputs[i],model)
        model=back_propogation(n_cells,inputs[i],outputs[i],model)
    #Testing phase

iterator(int(input("Enter number of iterations ")))