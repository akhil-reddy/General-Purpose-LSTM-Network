'''Created and developed by Akhil K'''
'''Find me at blitzkrieg.akhil@gmail.com'''
import numpy as np
import math
import random

#Learning rate
lr=0.35
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
    f_t=[]
    i_t=[]
    del_c=[]
    o_t=[]
    h_t=0
    #Contains all c_t-1 values at each timestamp
    c_t=[]
    c_t[0]=0
    obt_out=[]
    #Contains all h_t-1 values at each timestamp
    pcell_t=[]
    for i in range(n_cells):
        f_t[i]=sigmoid(model['f_tw'][0]*h_t+model['f_tw'][1]*inputs[i]+model['f_tw'][2])
        i_t[i]=sigmoid(model['i_tw'][0]*h_t+model['i_tw'][1]*inputs[i]+model['i_tw'][2]) 
        del_c[i]=np.tanh(model['del_cw'][0]*h_t+model['del_cw'][1]*inputs[i]+model['del_cw'][2])
        pcell_t.append(h_t)
        #Because c_t[0]=0
        c_t[i+1]=f_t[i]*c_t[i]+i_t[i]*del_c[i]
        o_t[i]=sigmoid(model['o_tw'][0]*h_t+model['o_tw'][1]*inputs[i]+model['o_tw'][2])
        h_t=o_t[i]*tanh(c_t[i+1])
        obt_out.append(h_t)
    model['obt_out']=obt_out
    model['pcell_t']=pcell_t
    model['c_t']=c_t
    model['f_t']=f_t
    model['i_t']=i_t
    model['del_c']=del_c
    model['o_t']=o_t
    return model  

def back_propogation(n_cells,inputs,outputs,model):
    for i in range(n_cells):
        #To get index from the end
        #o_t weights updation
        model['o_tw'][-i-1][0]=model['o_tw'][-i-1][0]-lr*model['pcell_t'][-i-1]*(obt_out[-i-1]-outputs[-i-1])*np.tanh(model['c_t'][-i-1])
        model['o_tw'][-i-1][1]=model['o_tw'][-i-1][1]-lr*inputs[-i-1]*(obt_out[-i-1]-outputs[-i-1])*np.tanh(model['c_t'][-i-1])
        model['o_tw'][-i-1][2]=model['o_tw'][-i-1][2]-lr*(model['obt_out'][-i-1]-outputs[-i-1])*np.tanh(model['c_t'][-i-1])
        #f_t weights updation
        model['f_tw'][-i-1][0]=model['f_tw'][-i-1][0]-lr*model['pcell_t'][-i-1]*(obt_out[-i-1]-outputs[-i-1])*model['o_t'][-i-1]*(1-(np.tanh(model['c_t'][-i-1]))**2)*model['c_t'][-i-2]
        model['f_tw'][-i-1][1]=model['f_tw'][-i-1][1]-lr*inputs[-i-1]*(obt_out[-i-1]-outputs[-i-1])**model['o_t'][-i-1]*(1-(np.tanh(model['c_t'][-i-1]))**2)*model['c_t'][-i-2]
        model['f_tw'][-i-1][2]=model['f_tw'][-i-1][2]-lr*(obt_out[-i-1]-outputs[-i-1])*model['o_t'][-i-1]*(1-(np.tanh(model['c_t'][-i-1]))**2)*model['c_t'][-i-2]
        #i_t weights updation
        model['i_tw'][-i-1][0]=model['i_tw'][-i-1][0]-lr*model['pcell_t'][-i-1]*(obt_out[-i-1]-outputs[-i-1])*model['o_t'][-i-1]*(1-(np.tanh(model['c_t'][-i-1]))**2)*model['del_c'][-i-1]
        model['i_tw'][-i-1][1]=model['i_tw'][-i-1][1]-lr*inputs[-i-1]*(obt_out[-i-1]-outputs[-i-1])*model['o_t'][-i-1]*(1-(np.tanh(model['c_t'][-i-1]))**2)*model['del_c'][-i-1]
        model['i_tw'][-i-1][2]=model['i_tw'][-i-1][2]-lr*(obt_out[-i-1]-outputs[-i-1])*model['o_t'][-i-1]*(1-(np.tanh(model['c_t'][-i-1]))**2)*model['del_c'][-i-1]
        #del_c weights updation
        model['del_cw'][-i-1][0]=model['del_cw'][-i-1][0]-lr*model['pcell_t'][-i-1]*(obt_out[-i-1]-outputs[-i-1])*model['o_t'][-i-1]*(1-(np.tanh(model['c_t'][-i-1]))**2)*model['i_t'][-i-1]
        model['del_cw'][-i-1][1]=model['del_cw'][-i-1][1]-lr*inputs[-i-1]*(obt_out[-i-1]-outputs[-i-1])*model['o_t'][-i-1]*(1-(np.tanh(model['c_t'][-i-1]))**2)*model['i_t'][-i-1]
        model['del_cw'][-i-1][2]=model['del_cw'][-i-1][2]-lr*(obt_out[-i-1]-outputs[-i-1])*model['o_t'][-i-1]*(1-(np.tanh(model['c_t'][-i-1]))**2)*model['i_t'][-i-1]
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
            outputs[i].append(int(input("Enter an output or 0 for no output")))
        model=forward_propogation(n_cells,inputs[i],model)
        model=back_propogation(n_cells,inputs[i],outputs[i],model)
    #Testing phase

iterator(int(input("Enter number of iterations ")))