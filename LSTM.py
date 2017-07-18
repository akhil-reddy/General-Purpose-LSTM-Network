'''Created and developed by Akhil K'''
'''Find me at blitzkrieg.akhil@gmail.com'''
import numpy as np
import math

def init(units,n_inputs,n_outputs,nn_hidden):
    nodes=[]
    #Network between hidden layers of consecutive LSTM units
    iu_network=[]
    self_network=[]
    for i in range(units):
        
    return {'nodes':nodes,'iu_network':iu_network,'self_network':self_network}
def forward_propogation(model,inputs):
    nodes=model['nodes']
    iu_network=model['iu_network']
    self_network=model['self_network']
    return nodes

def back_propogation(model,inputs,outputs):
    nodes=model['nodes']
    iu_network=model['iu_network']
    self_network=model['self_network']
    return {'nodes':nodes,'iu_network':iu_network,'self_network':self_network}

def iterator(epochs):
    #Training phase
    inputs=[]
    outputs=[]
    units=int(input("Enter the number of LSTM units required  "))
    n_inputs=int(input("Enter the number of input nodes (>0 of course) "))
    n_outputs=int(input("Enter the number of output nodes (>0 of course) "))
    nn_hidden=int(input("Enter the memory required (number of hidden layer nodes) "))
    model=init(units,n_inputs,n_outputs,nn_hidden)
    for i in range(units):
        if n_inputs==1:
            inputs.append(int(input("Enter your input")))
        else:
            inputs.append([])
            for j in range(n_inputs):
                inputs[i].append(int(input("Enter your input")))
    for i in range(units):
        if n_outputs==1:
            outputs.append(int(input("Enter your output")))
        else:
            outputs.append([])
            for j in range(n_outputs):
                outputs[i].append(int(input("Enter your output")))
    for i in range(epochs):
        nodes=forward_propogation(model,inputs)
        model['nodes']=nodes
        model=back_propogation(model,inputs,outputs)
    #Testing phase

iterator(int(input("Enter number of epochs")))