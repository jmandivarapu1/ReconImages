
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import sys



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x1 = F.relu(F.max_pool2d(self.conv1(x), 2))
        x2 = F.relu(F.max_pool2d(self.conv2(x1), 2))
        x22 = x2.view(-1, 320)
        x3 = F.relu(self.fc1(x22))
        # x = F.dropout(x, training=self.training)
        x4 = self.fc2(x3)
        return F.log_softmax(x4, dim=1), (x1, x2, x3, x4)

class variableWidthNet(nn.Module):
    def __init__(self, hiddenDim):
        super(variableWidthNet, self).__init__()
        self.fc1 = nn.Linear(1, hiddenDim)  # 45  65 ORIGINALLY 
        self.fc2 = nn.Linear(hiddenDim, 1)

    def forward(self, x):
        x1 = F.tanh(self.fc1(x))
        x2 = self.fc2(x1)
        return x2

class variableDepthNet(nn.Module):
    def __init__(self, hiddenDims, training):
        super(variableDepthNet, self).__init__()
        self.training = training
        if len(hiddenDims) == 2:
            self.hiddenLayers = 2
            self.fc1 = nn.Linear(1, hiddenDims[0])
            self.fc2 = nn.Linear(hiddenDims[0], hiddenDims[1])
            self.fc3 = nn.Linear(hiddenDims[1], 1)
        if len(hiddenDims) == 3: 
            self.hiddenLayers = 3
            self.fc1 = nn.Linear(1, hiddenDims[0])
            self.fc2 = nn.Linear(hiddenDims[0], hiddenDims[1])
            self.fc3 = nn.Linear(hiddenDims[1], hiddenDims[2])
            self.fc4 = nn.Linear(hiddenDims[2], 1)
        if len(hiddenDims) == 4:
            self.hiddenLayers = 4
            self.fc1 = nn.Linear(1, hiddenDims[0])
            self.fc2 = nn.Linear(hiddenDims[0], hiddenDims[1])
            self.fc3 = nn.Linear(hiddenDims[1], hiddenDims[2])
            self.fc4 = nn.Linear(hiddenDims[2], hiddenDims[3])
            self.fc5= nn.Linear(hiddenDims[3], 1)
        if len(hiddenDims) == 5:
            self.hiddenLayers = 5
            self.fc1 = nn.Linear(1, hiddenDims[0])
            self.fc2 = nn.Linear(hiddenDims[0], hiddenDims[1])
            self.fc3 = nn.Linear(hiddenDims[1], hiddenDims[2])
            self.fc4 = nn.Linear(hiddenDims[2], hiddenDims[3])
            self.fc5= nn.Linear(hiddenDims[3], hiddenDims[4])
            self.fc6 = nn.Linear(hiddenDims[4], 1)
        if len(hiddenDims) == 6:
            self.hiddenLayers = 6
            self.fc1 = nn.Linear(1, hiddenDims[0])
            self.fc2 = nn.Linear(hiddenDims[0], hiddenDims[1])
            self.fc3 = nn.Linear(hiddenDims[1], hiddenDims[2])
            self.fc4 = nn.Linear(hiddenDims[2], hiddenDims[3])
            self.fc5= nn.Linear(hiddenDims[3], hiddenDims[4])
            self.fc6 = nn.Linear(hiddenDims[4], hiddenDims[5])
            self.fc7 = nn.Linear(hiddenDims[5], 1)
        

    def forward(self, x):
        if self.hiddenLayers == 2:
            x1 = F.tanh(self.fc1(x))
            x1 = F.dropout(x1,p=0.25, training=self.training)
            x2 = F.tanh(self.fc2(x1))
            x2 = F.dropout(x2,p=0.25, training=self.training)
            x3 = self.fc3(x2)
            return x3
        if self.hiddenLayers == 3:
            x1 = F.tanh(self.fc1(x))
            x1 = F.dropout(x1,p=0.25, training=self.training)
            x2 = F.tanh(self.fc2(x1))
            x2 = F.dropout(x2,p=0.25, training=self.training)
            x3 = F.tanh(self.fc3(x2))
            x2 = F.dropout(x2,p=0.25, training=self.training)
            x4 = self.fc4(x3)
            return x4
        if self.hiddenLayers == 4:
            x1 = F.tanh(self.fc1(x))
            x1 = F.dropout(x1,p=0.25, training=self.training)
            x2 = F.tanh(self.fc2(x1))
            x2 = F.dropout(x2,p=0.25, training=self.training)
            x3 = F.tanh(self.fc3(x2))
            x3 = F.dropout(x3,p=0.25, training=self.training)
            x4 = F.tanh(self.fc4(x3))
            x4 = F.dropout(x4,p=0.25, training=self.training)
            x5 = self.fc5(x4)
            return x5
        if self.hiddenLayers == 5:
            x1 = F.tanh(self.fc1(x))
            x1 = F.dropout(x1,p=0.25, training=self.training)
            x2 = F.tanh(self.fc2(x1))
            x2 = F.dropout(x2,p=0.25, training=self.training)
            x3 = F.tanh(self.fc3(x2))
            x3 = F.dropout(x3,p=0.25, training=self.training)
            x4 = F.tanh(self.fc4(x3))
            x4 = F.dropout(x4,p=0.25, training=self.training)
            x5 = F.tanh(self.fc5(x4))
            x5 = F.dropout(x5,p=0.25, training=self.training)
            x6 = self.fc6(x5)
            return x6
        if self.hiddenLayers == 6:
            x1 = F.tanh(self.fc1(x))
            x1 = F.dropout(x1,p=0.25, training=self.training)
            x2 = F.tanh(self.fc2(x1))
            x2 = F.dropout(x2,p=0.25, training=self.training)
            x3 = F.tanh(self.fc3(x2))
            x3 = F.dropout(x3,p=0.25, training=self.training)
            x4 = F.tanh(self.fc4(x3))
            x4 = F.dropout(x4,p=0.25, training=self.training)
            x5 = F.tanh(self.fc5(x4))
            x5 = F.dropout(x5,p=0.25, training=self.training)
            x6 = F.tanh(self.fc6(x5))
            x6 = F.dropout(x6,p=0.25, training=self.training)
            x7 = self.fc7(x6)
            return x7


class variableDepthNet_v2(nn.Module):
     def __init__(self, hiddenDims):
        super(variableDepthNet_v2, self).__init__()
        self.fc1 = nn.Linear(1, hiddenDims[0])
        
        # self.hiddenLayers = {}
        
        # for h in range(len(hiddenDims)-1):
        #     self.hiddenLayers[h] = nn.Linear(hiddenDims[h], hiddenDims[h+1]).cpu()  
    
        self.hidden = nn.ModuleList()
        for k in range(len(hiddenDims)-1):
            self.hidden.append(nn.Linear(hiddenDims[k], hiddenDims[k+1]))

        self.fc2 = nn.Linear(hiddenDims[len(hiddenDims)-1], 1)

        # self.hidden = nn.ModuleList()
        # for k in range(len(hiddenDims)-1):
        #     self.hidden.append(nn.Linear(hiddenDims[k], hiddenDims[k+1]))

     def forward(self, x):
        x1 = F.leaky_relu(self.fc1(x))
        for h in range(len(self.hidden)):
            x1 = F.leaky_relu(self.hidden[h](x1))
            # x1 = F.dropout(x1,p=0.15)
        x1 = self.fc2(x1)
        return x1 

class variableDepthNet_v3(nn.Module):
     def __init__(self, inSize, outSize,hiddenDims):
        super(variableDepthNet_v3, self).__init__()
        self.fc1 = nn.Linear(inSize, hiddenDims[0])
        
        # self.hiddenLayers = {}
        
        # for h in range(len(hiddenDims)-1):
        #     self.hiddenLayers[h] = nn.Linear(hiddenDims[h], hiddenDims[h+1]).cpu()  
    
        self.hidden = nn.ModuleList()
        for k in range(len(hiddenDims)-1):
            self.hidden.append(nn.Linear(hiddenDims[k], hiddenDims[k+1]))

        self.fc2 = nn.Linear(hiddenDims[len(hiddenDims)-1], outSize)

        # self.hidden = nn.ModuleList()
        # for k in range(len(hiddenDims)-1):
        #     self.hidden.append(nn.Linear(hiddenDims[k], hiddenDims[k+1]))

     def forward(self, x):
        x1 = F.leaky_relu(self.fc1(x))
        for h in range(len(self.hidden)):
            x1 = F.leaky_relu(self.hidden[h](x1))
            # x1 = F.dropout(x1,p=0.15)
        x1 = self.fc2(x1)
        return x1 

class META_LINEAR_MODEL(nn.Module):
     def __init__(self, inputDim, outputDim):
        super(META_LINEAR_MODEL, self).__init__()
        self.fc1 = nn.Linear(inputDim, 45)  # 45  65 ORIGINALLY 
        self.fc2 = nn.Linear(45, 65)
        self.fc3 = nn.Linear(65, 45)
        self.fc4 = nn.Linear(45, outputDim)

     def forward(self, x):
        x1 = F.tanh(self.fc1(x))
        x2 = F.tanh(self.fc2(x1))
        x3 = F.tanh(self.fc3(x2))
        x4 = self.fc4(x3)
        return x4 

class META_LINEAR_MODEL_2(nn.Module):
     def __init__(self, inputDim, outputDim):
        super(META_LINEAR_MODEL_2, self).__init__()
        self.fc1 = nn.Linear(inputDim, 55)
        self.fc2 = nn.Linear(55, 100)
        self.fc3 = nn.Linear(100, 55)
        self.fc4 = nn.Linear(55, outputDim)

     def forward(self, x):
        x1 = F.tanh(self.fc1(x))
        x2 = F.tanh(self.fc2(x1))
        x3 = F.tanh(self.fc3(x2))
        x4 = self.fc4(x3)
        return x4 


class metaAE(nn.Module):
    def __init__(self, inputDim, outputDim):
        super(metaAE, self).__init__()
        self.fc1 = nn.Linear(inputDim, 15)
        self.fc2 = nn.Linear(15, outputDim)
       
    def encoder(self, x):
        h1 = self.fc1(x)
        return h1

    def decoder(self, z):
        h2 = self.fc2(z)
        return h2

    def forward(self, x):
        h1 = self.encoder(x)
        h2 = self.decoder(h1)
        return h2

class metaAE3(nn.Module):
    def __init__(self, inputDim, outputDim):
        super(metaAE3, self).__init__()
        self.fc1 = nn.Linear(inputDim, 2000)
        self.fc2 = nn.Linear(2000, 800)
        self.fc3 = nn.Linear(800, 2000)
        self.fc4 = nn.Linear(2000, outputDim)
       
    def encoder(self, x):
        h1 = F.tanh(self.fc1(x))
        h2 = self.fc2(h1)
        return h2

    def decoder(self, z):
        h3 = F.tanh(self.fc3(z))
        h4 = self.fc4(h3)
        return h4

    def forward(self, x):
        h1 = self.encoder(x)
        h2 = self.decoder(h1)
        return h2
        
class RegressionAgent(nn.Module):
    def __init__(self):
        super(RegressionAgent, self).__init__()
        self.fc1 = nn.Linear(40, 40)
        self.fc2 = nn.Linear(40, 40)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x1)
        return x2

class metaModel_CR(nn.Module):
     def __init__(self, inputDim, outputDim):
        super(RegressionAgent, self).__init__()
        self.fc1 = nn.Linear(inputDim, 40)
        self.fc2 = nn.Linear(40, outputDim)

     def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x1)
        return x2

class CAE_91(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.fc1 = nn.Linear(21432, 2000)
        self.fc2 = nn.Linear(2000,200)
        self.fc3 = nn.Linear(200,2000)
        self.fc4 = nn.Linear(2000, 21432)
        self.sigmoid = nn.Tanh()

    def encoder(self, x):
        h1 = self.sigmoid(self.fc1(x.view(-1, 21432)))
        return self.sigmoid(self.fc2(h1))

    def decoder(self, z):
        h2 = self.sigmoid(self.fc3(z))
        return self.sigmoid(self.fc4(h2))

    def forward(self, x):
        h1 = self.encoder(x)
        h2 = self.decoder(h1)
        return h1, h2


class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.fc1 = nn.Linear(21432, 5)

        self.fc2 = nn.Linear(5, 21432)
        self.relu = nn.ELU()
        self.sigmoid = nn.ELU()

    def encoder(self, x):
        h1 = self.fc1(x.view(-1, 21432))
        return h1

    def decoder(self, z):
        h2 = (self.fc2(z))
        return h2

    def forward(self, x):
        h1 = self.encoder(x)
        h2 = self.decoder(h1)
        return h1, h2

class CAE_multiLayer(nn.Module):
    def __init__(self):
        super(CAE_multiLayer, self).__init__()
        self.fc1 = nn.Linear(21432, 4)
        self.fc11 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 4)
        self.fc22 = nn.Linear(4, 21432)
        self.elu = nn.ELU()


    def encoder(self, x):
        h1 = self.elu(self.fc1(x.view(-1, 21432)))
        h2 = self.fc11(h1)
        return h2

    def decoder(self, z):
        h3 = self.elu(self.fc2(z))
        h4 = self.fc22(h3)
        return h4

    def forward(self, x):
        h1 = self.encoder(x)
        h2 = self.decoder(h1)
        return h1, h2



"""
###########################################################################################
    (START) DAN-NET BUILDING BLOCKS
###########################################################################################
"""
#-----------------------------------------------------------------------------
class Single_Synapse(nn.Module):
    """
        A Single_Synapse is an MLP which distills n_synaptic channels into a single value 
        traversing the connection.
    """
    def __init__(self, n_channels):
        super(Single_Synapse, self).__init__()
        self.fc1 = nn.Linear(n_channels, 3)  
        self.fc2 = nn.Linear(3, 1) 
    def forward(self, x):
        x1 = torch.tanh(self.fc1(x))
        x2 = torch.tanh(self.fc2(x1))
        return x2 

#-----------------------------------------------------------------------------
class Node_Aggregator(nn.Module):
    """
        A Node_Aggregator is an MLP which distills the filtered signals arriving on num_inbound 
        connections into a single outbound activation scalar. 
    """
    def __init__(self, num_inbound):
        super(Node_Aggregator, self).__init__()
        self.fc1 = nn.Linear(num_inbound, 3)  
        self.fc2 = nn.Linear(3, 1) 
    def forward(self, x): 
        x1 = torch.tanh(self.fc1(x))
        x2 = self.fc2(x1)
        return x2 
#-----------------------------------------------------------------------------
class DAN(nn.Module):
    """
        A DAN is composed of num_inbound Single_Synapses and a single Node_Aggregator 
            num_inbound:  number of inbound connections
            n_channels:  number of synaptic channels per connection 
            
    """
    def __init__(self, num_inbound, n_channels, bias_vector_size=None, output_node=None):
        super(DAN, self).__init__()
       
        # Synapses ###########
        self.synapses = nn.ModuleDict()
        self.output_node = output_node
        self.num_inbound = num_inbound
        for n in range(num_inbound):
            self.synapses[str(n)] = Single_Synapse(n_channels)

        # Node Aggregator ##########
        if bias_vector_size is not None:
            self.node_aggregator = Node_Aggregator(num_inbound+bias_vector_size)
        else:
            self.node_aggregator = Node_Aggregator(num_inbound)

    def forward(self, x, bias_vector=None):
        # pass filtered input through the corresponding synapse 
        self.synapse_activations = torch.Tensor(np.zeros(self.num_inbound))
        i = 0
        
        for synapse in self.synapses:
            self.synapse_activations[i] = self.synapses[synapse](x[i])
            i+=1
        # concatenate with the node location to bias the node & ...
        # pass the synaptic outputs through the node aggregator 
        if bias_vector is not None:
            out = self.node_aggregator(torch.cat((self.synapse_activations, bias_vector)))
        else:  
            out = self.node_aggregator(self.synapse_activations)
        # if self.output_node == False: 
        #     out = torch.tanh(out)
        return out

class Simple_DAN(nn.Module):
    def __init__(self, n_channels):
        super(Simple_DAN, self).__init__()
        self.dendrites = nn.Linear(n_channels, 15)  
        self.fc2 = nn.Linear(15, 8) 
        self.fc3 = nn.Linear(8, 1) 
    def forward(self, x): 
        x1 = F.leaky_relu(self.dendrites(x))
        x2 = F.leaky_relu(self.fc2(x1))
        x3 = self.fc3(x2)
        return x3 

#-----------------------------------------------------------------------------
class Single_VECs(nn.Module):
    """
        A VECS splits input along n synaptic channels 
        
    """
    def __init__(self, n_channels):
        super(Single_VECs, self).__init__()
        self.fc1 = nn.Linear(1, n_channels)  
    def forward(self, x):
        x1 = torch.tanh(self.fc1(x))
        return x1 
#-----------------------------------------------------------------------------
class NETWORK_OF_DANS_and_VECS(nn.Module):
    """
        params:
            dims (array, ints): array of layer widths 
            n_channels (int):  synaptic channels per connection 
            VECS_layers (array): array of ModuleDicts, which hold the VECs (single synapses) for that layer
            DAN_layers (array): array of ModuleDicts which hold the DANs for that layer  

        (1) we need to instantiate a VECS from every presynaptic node to every postsynaptic node. 
        (2) we will therefore have a layer of VECS
        (3) followed by a layer of DANs
    """
    def __init__(self, dims, n_channels):
        super(NETWORK_OF_DANS_and_VECS, self).__init__()
        self.dims = dims 
        self.n_channels = n_channels 
        self.VECS_layers = nn.ModuleList() #nn.ModuleDict() # container (dictionary) for the layers of VECS
        self.DAN_topologies = []
        
        # # node bias ------------------------------------
        self.layer_codes = np.eye(len(dims))[range(len(dims))]
        self.node_codes = []
        for layer in range(len(dims)):
            layer_code = self.layer_codes[layer]
            for node_idx in range(dims[layer]):
                node_code = np.eye(np.max(dims))[node_idx]
                self.node_codes.append(node_code)
        self.layer_codes = torch.Tensor(self.layer_codes)
        self.node_codes = torch.Tensor(self.node_codes)
        self.bias_vector_size = len(torch.cat((self.layer_codes[0], self.node_codes[0])))
         
        # construct fully-connected layers of VECS -------------------------------------------
        for layer_idx in range(len(dims)-1):   #for each layer, instantiate a layer of VECs
            VECs = nn.ModuleDict()
            for pre_node in range(dims[layer_idx]):
                for post_node in range(dims[layer_idx+1]):
                    VECs["pre_"+str(pre_node)+"_post_"+str(post_node)] = Single_VECs(n_channels)
            self.VECS_layers.append(VECs)
        
        # construct layers of DANs ----------------------------------------------------
        self.DAN_layers = nn.ModuleList()
        for layer_idx, dim in enumerate(dims):
            layer_DANs = nn.ModuleDict()
            if layer_idx > 0:
                for node_idx in range(dim):
                    torch.manual_seed(9)   # initialize all DANs to same parameters 
                    if node_idx == 0:
                        self.DAN_topologies.append([dims[layer_idx-1], n_channels, self.bias_vector_size])
                    if layer_idx == len(self.dims)-1:
                        output_node = True 
                    else: 
                        output_node = False
                    layer_DANs["node_"+str(node_idx)] = DAN(num_inbound=dims[layer_idx-1], n_channels=n_channels, bias_vector_size=self.bias_vector_size, output_node=output_node)  
                self.DAN_layers.append(layer_DANs)

    def forward(self, x):
        self.layer_activations = {}
        # input layer is a special case, no VECs filtering necessary
        for node in range(self.dims[0]):
            self.layer_activations["layer_0_node_"+str(node)] = x[node].view(-1,1) 
        
        # get the filtered input for each synapse on each DAN in the next layer -------------------- 
        for layer_idx in range(len(self.dims)-1):
            inbound_activation_vectors = torch.Tensor(np.zeros((self.dims[layer_idx+1], self.dims[layer_idx], self.n_channels))) # of size num_inbound_connections * n_channels 
           
            for node in range(self.dims[layer_idx+1]):  # for each post-synaptic node 
                # for each synapse on that post-synaptic node, equals number of ...
                # ... inbound connections == number of nodes in pre layer
                self.layer_activations["layer_"+str(layer_idx+1)+"_node_"+str(node)] = []
                for synapse in range(self.dims[layer_idx]):
                    node_key = "node_"+str(node)
                    synapse_key = str(synapse) 
                    VECs_key = "pre_"+str(synapse_key)+"_post_"+str(node)
                    #get the filtered activation for that synapse, will be of size n_channels 
                    filtered_act = self.VECS_layers[layer_idx][VECs_key](self.layer_activations["layer_"+str(layer_idx)+"_node_"+str(synapse)]) #clone not needed
                    inbound_activation_vectors[node][synapse] = filtered_act #clone not needed
                """
                    Once we have our filtered activations for each node, we need to pass them through the correct DAN for that layer, and 
                    collect the output. which will itself be filtered by VECS in the next layer. 
                """ 
                # node bias ------------------------------------
                bias_vector = torch.cat((self.layer_codes[layer_idx+1], self.node_codes[node]))
                # pass the synapse activations through the node
                node_out = self.DAN_layers[layer_idx]["node_"+str(node)](inbound_activation_vectors[node].clone(), bias_vector=bias_vector) #CLONE NEEDED
                self.layer_activations["layer_"+str(layer_idx+1)+"_node_"+str(node)] = node_out.view(-1,1) #clone not needed 
        net_out = torch.Tensor()
        for node in range(self.dims[-1]):
            net_out = torch.cat((self.layer_activations["layer_"+str(len(self.dims)-1)+"_node_"+str(node)], net_out)) #clone not needed
        return net_out
        
    def avg_DAN_grads(self):
        for layer_idx, dan_layer in enumerate(self.DAN_layers):
            mu_DAN = DAN(num_inbound=self.DAN_topologies[layer_idx][0], n_channels=self.n_channels, bias_vector_size=self.DAN_topologies[layer_idx][2])
            for name, p in mu_DAN.named_parameters():
                p.data.fill_(0)
            for node in dan_layer:
                for (node_paramName, node_param), (muDAN_paramName, mu_param) in zip(dan_layer[node].named_parameters(),mu_DAN.named_parameters()):
                    mu_param.data = mu_param.data + node_param.grad
            for name, p in mu_DAN.named_parameters():
                p.data = p.data / len(dan_layer)
            for node in dan_layer:
                for (node_paramName, node_param), (muDAN_paramName, mu_param) in zip(dan_layer[node].named_parameters(),mu_DAN.named_parameters()):
                    
                    # print("pre", node_paramName, node_param.grad)
                    node_param.grad.data = mu_param

    def avg_DANs(self):
        for layer_idx, dan_layer in enumerate(self.DAN_layers):
            mu_DAN = DAN(num_inbound=self.DAN_topologies[layer_idx][0], n_channels=self.n_channels, bias_vector_size=self.DAN_topologies[layer_idx][2])
            for name, p in mu_DAN.named_parameters():
                p.data.fill_(0)
            for node in dan_layer:
                for (node_paramName, node_param), (muDAN_paramName, mu_param) in zip(dan_layer[node].named_parameters(),mu_DAN.named_parameters()):
                    mu_param.data = mu_param.data + node_param
            for name, p in mu_DAN.named_parameters():
                p.data = p.data / len(dan_layer)
            for node in dan_layer:
                for (node_paramName, node_param), (muDAN_paramName, mu_param) in zip(dan_layer[node].named_parameters(),mu_DAN.named_parameters()):
                    
                    # print("pre", node_paramName, node_param.grad)
                    node_param.data = mu_param
                    
                    # print("post", node_paramName, node_param.grad)
                    # print("mu_param",  mu_param)
                        
            #     print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")   
            # print("############################################################")
        # sys.exit()
#-----------------------------------------------------------------------------
class NETWORK_OF_DANS(nn.Module):
    """
        params:
            dims (array of ints): array of layer widths.  Must be multiples of n_channels
            n_channels (int):  denotes input dimensions for DANs 
            DAN (MLP):  Simple_DAN, distibuted genome 
            VECS_layers (dict): ModuleDicts which hold the adapt parameters for that layer

        (1) we need to instantiate a fully-connected layer from the presynaptic layer (layer_idx) to the 
            post-synaptic layer (layer_idx+1)
        (2) we will therefore have a layer of VECS (theta)
        (3) followed by a layer of DANs (which themselves are multi-layer MLPs)
            - in practice, each layer has but a single DAN, which is distributed throughout the layer 
    """
    def __init__(self, dims, n_channels, genome=None):
        super(NETWORK_OF_DANS, self).__init__()   
        self.dims = dims 
        self.n_channels = n_channels 
        self.VECS_layers = nn.ModuleDict() #nn.ModuleDict() # container (dictionary) for the layers of VECS
        self.Skip_layers = nn.ModuleDict()
        #self.DAN_topologies = []
        #self.DAN_layers = nn.ModuleDict()#Simple_DAN(self.n_channels)
        self.layer_bias_codes = torch.Tensor(np.eye(len(self.dims)-1)).cpu()
        if genome is not None:
            self.DAN = genome
        else:
            self.DAN = Simple_DAN(n_channels+len(self.layer_bias_codes[0]))
        self.dan_acts = torch.Tensor().cpu()

        self.DAN_out = Simple_DAN(n_channels+len(self.layer_bias_codes[0]))

   

        self.Skip_layers["0_2"] = nn.Linear(1, self.dims[2]*self.n_channels)
        self.Skip_layers["1_3"] = nn.Linear(self.dims[1], self.n_channels)

        # construct fully-connected layers of VECS -------------------------------------------
        for l in range(len(self.dims)-1):
            self.VECS_layers["layer_"+str(l)] = nn.Linear(self.dims[l], self.dims[l+1]*n_channels)
            #if we want sparse VECS
            # if l > 0:
            #     i = 0
            #     for p in self.VECS_layers["layer_"+str(l)].parameters():
            #         if i == 0:
            #             mask = np.zeros(tuple(p.t().shape))
            #             for idx, row in enumerate(mask):
            #                 mask[idx][idx::self.n_channels] = 1
            #             mask = mask.transpose()
                        
            #             p.data = p * torch.Tensor(mask) 
            #         i+=1

        #construct a DAN for each layer -------------------------------------------
        # for l in range(len(self.dims)-1):
        #     self.DAN_layers["layer_"+str(l)] = Simple_DAN(n_channels)

    # if we want sparse VECS
    def maskVECS(self):
        for l in range(len(self.dims)-1):
            if l > 0:
                i = 0
                for p in self.VECS_layers.Standard_layers["layer_"+str(l)].parameters():
                    if i == 0:
                        mask = np.zeros(tuple(p.t().shape))
                        for idx, row in enumerate(mask):
                            mask[idx][idx::self.n_channels] = 1
                        mask = mask.transpose()
                        p.data = p * torch.Tensor(mask) 
                    i+=1

    def forward(self, x):
        hidden_acts = x
        #node_idx = 0
        layer_idx = 0
        self.skip_acts_0_2 = torch.Tensor().cpu()
        self.skip_acts_1_3 = torch.Tensor().cpu()
        for layer_key in self.VECS_layers:
            #print("hidden acts pre", hidden_acts)
            # print("-----------------------")
            # for p in self.VECS_layers[layer_key].parameters():

            #     print(p)
            #     print(p.shape)
            # print("--------------------------")
            
            hidden_acts_copy = hidden_acts.clone().detach()
            hidden_acts = self.VECS_layers[layer_key](hidden_acts) 
            
            if layer_idx == 1:
                self.skip_acts_0_2 = self.Skip_layers["0_2"](x)
                self.skip_acts_1_3 = self.Skip_layers["1_3"](hidden_acts_copy)
                hidden_acts = hidden_acts.add_(self.skip_acts_0_2)
            if layer_idx == 2:
                hidden_acts = hidden_acts.add_(self.skip_acts_1_3)
            
            #hidden_acts = torch.tanh(hidden_acts)
            
            #slice the hidden activations for the layer into sub-vectors to be passed through the DAN
            self.dan_acts = torch.Tensor().cpu()
            node_idx_ext = 0
            for z in [hidden_acts[0][node_idx:node_idx + self.n_channels] for node_idx in range(0, len(hidden_acts[0]), self.n_channels)]:
                #concat the output of each DAN in the layer to form the activation vector for the layer of DANs
                #print("node_idx", node_idx)
                if layer_idx == len(self.dims)-1:
                    #self.dan_acts = torch.cat((self.dan_acts, self.DAN_layers[layer_key](z)))    
                    z = torch.cat((z,self.layer_bias_codes[layer_idx]))
                    self.dan_acts = torch.cat((self.dan_acts, self.DAN(z))) 
                else:  
                    #self.dan_acts = torch.cat((self.dan_acts, torch.tanh(self.DAN_layers[layer_key](z)))) 
                    z = torch.cat((z,self.layer_bias_codes[layer_idx]))
                    self.dan_acts = torch.cat((self.dan_acts, torch.tanh(self.DAN(z))))
                node_idx_ext += 1
            #sys.exit()
            #assign as hidden_acts to be passed through the next layer of VECS
            hidden_acts = self.dan_acts.view(1, len(self.dan_acts)) 
            layer_idx +=1
        
        return hidden_acts

###########################################################################################################
class NETWORK_OF_DANS_batch(nn.Module):
    """
        params:
            dims (array of ints): array of layer widths.  Must be multiples of n_channels
            n_channels (int):  denotes input dimensions for DANs 
            DAN (MLP):  Simple_DAN, distibuted genome 
            VECS_layers (dict): ModuleDicts which hold the adapt parameters for that layer

        (1) we need to instantiate a fully-connected layer from the presynaptic layer (layer_idx) to the 
            post-synaptic layer (layer_idx+1)
        (2) we will therefore have a layer of VECS (theta)
        (3) followed by a layer of DANs (which themselves are multi-layer MLPs)
            - in practice, each layer has but a single DAN, which is distributed throughout the layer 
    """
    def __init__(self, dims, n_channels, batch_size=1,genome=None,):
        super(NETWORK_OF_DANS_batch, self).__init__()   
        self.dims = dims 
        self.batch_size = batch_size
        self.num_DAN_layers = len(dims)-1
        self.n_channels = n_channels 
        self.VECS_layers = nn.ModuleDict() #nn.ModuleDict() # container (dictionary) for the layers of VECS
        self.Skip_layers = nn.ModuleDict()
        #self.DAN_topologies = []


        #self.DAN_layers = nn.ModuleDict()#Simple_DAN(self.n_channels)
        self.layer_bias_codes = torch.Tensor(np.eye(len(self.dims)-1)).cpu()
        self.bias_codes = torch.zeros((self.num_DAN_layers,self.batch_size,self.num_DAN_layers))
        for l_idx in range(len(self.layer_bias_codes)):
            for y in range(self.batch_size):
                bias_zeros = torch.zeros((self.batch_size, len(self.layer_bias_codes[0])))
                for z_idx in range(len(bias_zeros)):
                    bias_zeros[z_idx] = self.layer_bias_codes[l_idx]
                
                self.bias_codes[l_idx] = bias_zeros
        self.layer_bias_codes = self.bias_codes
       
        if genome is not None:
            self.DAN = genome
        else:
            self.DAN = Simple_DAN(n_channels+len(self.layer_bias_codes[0][0]))
        self.dan_acts = torch.Tensor().cpu()

        

   

        self.Skip_layers["0_2"] = nn.Linear(self.dims[0], self.dims[2]*self.n_channels)
        self.Skip_layers["1_3"] = nn.Linear(self.dims[1], self.dims[3]*self.n_channels)

        # construct fully-connected layers of VECS -------------------------------------------
        for l in range(len(self.dims)-1):
            self.VECS_layers["layer_"+str(l)] = nn.Linear(self.dims[l], self.dims[l+1]*n_channels)

            # #for unique
            # for node_idx in range(self.dims[l+1]):
            #     self.DAN_layers["layer_"+str(l)+"_node_"+str(node_idx)] = Simple_DAN(n_channels+len(self.layer_bias_codes[0]))
          

    def forward(self, x):
        hidden_acts = x
        #node_idx = 0
        layer_idx = 0
        self.skip_acts_0_2 = torch.Tensor().cpu()
        self.skip_acts_1_3 = torch.Tensor().cpu()
        for layer_key in self.VECS_layers:
          
            
            hidden_acts_copy = hidden_acts.clone().detach()
            hidden_acts = self.VECS_layers[layer_key](hidden_acts) 

            
            
            if layer_idx == 1:
                

                self.skip_acts_0_2 = self.Skip_layers["0_2"](x)
                self.skip_acts_1_3 = self.Skip_layers["1_3"](hidden_acts_copy)
                hidden_acts = hidden_acts.add_(self.skip_acts_0_2)
            if layer_idx == 2:
                hidden_acts = hidden_acts.add_(self.skip_acts_1_3)
            
            #hidden_acts = torch.tanh(hidden_acts)
            
            #slice the hidden activations for the layer into sub-vectors to be passed through the DAN
            self.dan_acts = torch.Tensor().cpu()
            node_idx_ext = 0
            
            #print("hidden_acts", hidden_acts.size())
            for z in [hidden_acts[:,node_idx:node_idx + self.n_channels] for node_idx in range(0, len(hidden_acts[0]), self.n_channels)]:
                #concat the output of each DAN in the layer to form the activation vector for the layer of DANs
                  
                if layer_idx == len(self.dims)-1:
                    #self.dan_acts = torch.cat((self.dan_acts, self.DAN_layers[layer_key](z)))    
                    z = torch.cat((z,self.layer_bias_codes[layer_idx]), dim=1)
                    self.dan_acts = torch.cat((self.dan_acts, self.DAN(z)), dim=1) 
                else:   
                    #self.dan_acts = torch.cat((self.dan_acts, torch.tanh(self.DAN_layers[layer_key](z)))) 
                    z = torch.cat((z,self.layer_bias_codes[layer_idx]), dim=1)
                    self.dan_acts = torch.cat((self.dan_acts, F.leaky_relu(self.DAN(z))), dim=1)
                    
                node_idx_ext += 1
            
            #assign as hidden_acts to be passed through the next layer of VECS
            
            hidden_acts = self.dan_acts.view(self.dan_acts.size()[0], self.dan_acts.size()[1]) 
            layer_idx +=1
        
        return hidden_acts
##########  End DANs batch #########################################################################


##########################################################################################################
class NETWORK_OF_DANS_perLayerGenomes(nn.Module):
    """
        params:
            dims (array of ints): array of layer widths.  Must be multiples of n_channels
            n_channels (int):  denotes input dimensions for DANs 
            DAN (MLP):  Simple_DAN, distibuted genome 
            VECS_layers (dict): ModuleDicts which hold the adapt parameters for that layer

        (1) we need to instantiate a fully-connected layer from the presynaptic layer (layer_idx) to the 
            post-synaptic layer (layer_idx+1)
        (2) we will therefore have a layer of VECS (theta)
        (3) followed by a layer of DANs (which themselves are multi-layer MLPs)
            - in practice, each layer has but a single DAN, which is distributed throughout the layer 
    """
    def __init__(self, dims, n_channels, genome=None):
        super(NETWORK_OF_DANS_perLayerGenomes, self).__init__()   
        self.dims = dims 
        self.n_channels = n_channels 
        self.VECS_layers = nn.ModuleDict() #nn.ModuleDict() # container (dictionary) for the layers of VECS
        self.Skip_layers = nn.ModuleDict()
        #self.DAN_topologies = []
        self.DAN_layers = nn.ModuleDict()#Simple_DAN(self.n_channels)
        self.layer_bias_codes = torch.Tensor(np.eye(len(self.dims)-1)).cpu()
        # if genome is not None:
        #     self.DAN = genome
        # else:
        #     self.DAN = Simple_DAN(n_channels+len(self.layer_bias_codes[0]))
        # self.dan_acts = torch.Tensor().cpu()

        # self.DAN_out = Simple_DAN(n_channels+len(self.layer_bias_codes[0]))

   

        self.Skip_layers["0_2"] = nn.Linear(1, self.dims[2]*self.n_channels)
        self.Skip_layers["1_3"] = nn.Linear(self.dims[1], self.n_channels)

        # construct fully-connected layers of VECS -------------------------------------------
        for l in range(len(self.dims)-1):
            self.VECS_layers["layer_"+str(l)] = nn.Linear(self.dims[l], self.dims[l+1]*n_channels)
            self.DAN_layers["layer_"+str(l)] = Simple_DAN(n_channels+len(self.layer_bias_codes[0]))
            #if we want sparse VECS
            # if l > 0:
            #     i = 0
            #     for p in self.VECS_layers["layer_"+str(l)].parameters():
            #         if i == 0:
            #             mask = np.zeros(tuple(p.t().shape))
            #             for idx, row in enumerate(mask):
            #                 mask[idx][idx::self.n_channels] = 1
            #             mask = mask.transpose()
                        
            #             p.data = p * torch.Tensor(mask) 
            #         i+=1

        #construct a DAN for each layer -------------------------------------------
        # for l in range(len(self.dims)-1):
        #     self.DAN_layers["layer_"+str(l)] = Simple_DAN(n_channels)

    # if we want sparse VECS
    def maskVECS(self):
        for l in range(len(self.dims)-1):
            if l > 0:
                i = 0
                for p in self.VECS_layers.Standard_layers["layer_"+str(l)].parameters():
                    if i == 0:
                        mask = np.zeros(tuple(p.t().shape))
                        for idx, row in enumerate(mask):
                            mask[idx][idx::self.n_channels] = 1
                        mask = mask.transpose()
                        p.data = p * torch.Tensor(mask) 
                    i+=1

    def forward(self, x):
        hidden_acts = x
        #node_idx = 0
        layer_idx = 0
        self.skip_acts_0_2 = torch.Tensor().cpu()
        self.skip_acts_1_3 = torch.Tensor().cpu()
        for layer_key in self.VECS_layers:
            #print("hidden acts pre", hidden_acts)
            # print("-----------------------")
            # for p in self.VECS_layers[layer_key].parameters():

            #     print(p)
            #     print(p.shape)
            # print("--------------------------")
            
            hidden_acts_copy = hidden_acts.clone().detach()
            hidden_acts = self.VECS_layers[layer_key](hidden_acts) 
            
            if layer_idx == 1:
                self.skip_acts_0_2 = self.Skip_layers["0_2"](x)
                self.skip_acts_1_3 = self.Skip_layers["1_3"](hidden_acts_copy)
                hidden_acts = hidden_acts.add_(self.skip_acts_0_2)
            if layer_idx == 2:
                hidden_acts = hidden_acts.add_(self.skip_acts_1_3)
            
            #hidden_acts = torch.tanh(hidden_acts)
            
            #slice the hidden activations for the layer into sub-vectors to be passed through the DAN
            self.dan_acts = torch.Tensor().cpu()
            node_idx_ext = 0
            for z in [hidden_acts[0][node_idx:node_idx + self.n_channels] for node_idx in range(0, len(hidden_acts[0]), self.n_channels)]:
                #concat the output of each DAN in the layer to form the activation vector for the layer of DANs
                #print("node_idx", node_idx)
                if layer_idx == len(self.dims)-1:
                    #self.dan_acts = torch.cat((self.dan_acts, self.DAN_layers[layer_key](z)))    
                    z = torch.cat((z,self.layer_bias_codes[layer_idx]))
                    self.dan_acts = torch.cat((self.dan_acts, self.DAN_layers[layer_key](z))) 
                else:  
                    #self.dan_acts = torch.cat((self.dan_acts, torch.tanh(self.DAN_layers[layer_key](z)))) 
                    z = torch.cat((z,self.layer_bias_codes[layer_idx]))
                    self.dan_acts = torch.cat((self.dan_acts, torch.tanh(self.DAN_layers[layer_key](z))))
                node_idx_ext += 1
            #sys.exit()
            #assign as hidden_acts to be passed through the next layer of VECS
            hidden_acts = self.dan_acts.view(1, len(self.dan_acts)) 
            layer_idx +=1
        
        return hidden_acts

class NETWORK_OF_DANS_uniqueNeurons(nn.Module):
    """
        params:
            dims (array of ints): array of layer widths.  Must be multiples of n_channels
            n_channels (int):  denotes input dimensions for DANs 
            DAN (MLP):  Simple_DAN, distibuted genome 
            VECS_layers (dict): ModuleDicts which hold the adapt parameters for that layer

        (1) we need to instantiate a fully-connected layer from the presynaptic layer (layer_idx) to the 
            post-synaptic layer (layer_idx+1)
        (2) we will therefore have a layer of VECS (theta)
        (3) followed by a layer of DANs (which themselves are multi-layer MLPs)
            - in practice, each layer has but a single DAN, which is distributed throughout the layer 
    """
    def __init__(self, dims, n_channels, genome=None):
        super(NETWORK_OF_DANS_uniqueNeurons, self).__init__()   
        self.dims = dims 
        self.n_channels = n_channels 
        self.VECS_layers = nn.ModuleDict() #nn.ModuleDict() # container (dictionary) for the layers of VECS
        self.Skip_layers = nn.ModuleDict()
        #self.DAN_topologies = []
        self.DAN_layers = nn.ModuleDict()#Simple_DAN(self.n_channels)
        self.layer_bias_codes = torch.Tensor(np.eye(len(self.dims)-1)).cpu()
        
        self.Skip_layers["0_2"] = nn.Linear(1, self.dims[2]*self.n_channels)
        self.Skip_layers["1_3"] = nn.Linear(self.dims[1], self.n_channels)

        # construct fully-connected layers of VECS -------------------------------------------
        for l in range(len(self.dims)-1):
            self.VECS_layers["layer_"+str(l)] = nn.Linear(self.dims[l], self.dims[l+1]*n_channels)
            for node_idx in range(self.dims[l+1]):
                self.DAN_layers["layer_"+str(l)+"_node_"+str(node_idx)] = Simple_DAN(n_channels+len(self.layer_bias_codes[0]))
        

    def forward(self, x):
        hidden_acts = x
        #node_idx = 0
        layer_idx = 0
        self.skip_acts_0_2 = torch.Tensor().cpu()
        self.skip_acts_1_3 = torch.Tensor().cpu()
        for layer_key in self.VECS_layers:
          
            
            hidden_acts_copy = hidden_acts.clone().detach()
            hidden_acts = self.VECS_layers[layer_key](hidden_acts) 
            
            if layer_idx == 1:
                self.skip_acts_0_2 = self.Skip_layers["0_2"](x)
                self.skip_acts_1_3 = self.Skip_layers["1_3"](hidden_acts_copy)
                hidden_acts = hidden_acts.add_(self.skip_acts_0_2)
            if layer_idx == 2:
                hidden_acts = hidden_acts.add_(self.skip_acts_1_3)
            
            #hidden_acts = torch.tanh(hidden_acts)
            
            #slice the hidden activations for the layer into sub-vectors to be passed through the DAN
            self.dan_acts = torch.Tensor().cpu()
            node_idx_ext = 0
            for z in [hidden_acts[0][node_idx:node_idx + self.n_channels] for node_idx in range(0, len(hidden_acts[0]), self.n_channels)]:
                #concat the output of each DAN in the layer to form the activation vector for the layer of DANs
                #print("node_idx", node_idx)
                if layer_idx == len(self.dims)-1:
                    #self.dan_acts = torch.cat((self.dan_acts, self.DAN_layers[layer_key](z)))    
                    z = torch.cat((z,self.layer_bias_codes[layer_idx]))
                    self.dan_acts = torch.cat((self.dan_acts, self.DAN_layers[layer_key+"_node_"+str(node_idx_ext)](z))) 
                else:  
                    #self.dan_acts = torch.cat((self.dan_acts, torch.tanh(self.DAN_layers[layer_key](z)))) 
                    z = torch.cat((z,self.layer_bias_codes[layer_idx]))
                    self.dan_acts = torch.cat((self.dan_acts, torch.tanh(self.DAN_layers[layer_key+"_node_"+str(node_idx_ext)](z))))
                node_idx_ext += 1
           
            #assign as hidden_acts to be passed through the next layer of VECS
            hidden_acts = self.dan_acts.view(1, len(self.dan_acts)) 
            layer_idx +=1
        
        return hidden_acts

