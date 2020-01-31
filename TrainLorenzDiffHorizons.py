#!/usr/bin/env python
# coding: utf-8

# Script to train various Networks to learn Lorenz model dynamics with different loss functions
# Script taken from Dueben and Bauer 2018 paper supplematary info and modified

from comet_ml import Experiment

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler


# Set up comet exp tracking
hyper_params = {
    "batch_size": 512,
    "num_epochs": 50,  # in D&B paper the NN's were trained for at least 200 epochs 
    "learning_rate": 0.001
}

#experiment = Experiment(project_name="LorenzTrainDiffHorizons")
experiment = Experiment()
experiment.log_parameters(hyper_params)

# run on GPUs if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)
print()

K = 8                   # 8 X variables in the Lorenz model
t_int = 0.005
#n_run = int(2000000/8)  # Want 2mill samples, and obtain 8 per time step sampled
n_run = 512     # Testing

##############################################################
#print('Read in input-output training pairs from text file') #
##############################################################

file_train = '/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/LORENZ_DATASETS/Lorenz_full.txt'

data_list_tm1 = []   # value at time minus 1
data_list_t = []     # value at current time step
data_list_tp10 = []  # value at time plus 10
data_list_tp100 = [] # value at time plus 100

file = open(file_train, 'r')
for skip in range(50):  # skip first 50 lines as initialisation is a bit odd
    a_str = file.readline()
for i in range(n_run):
    a_str = file.readline() ;  data_list_tm1.append(a_str.split())
    a_str = file.readline() ;  data_list_t.append(a_str.split())
    for j in range(8):  # skip 8 lines
       a_str = file.readline()
    a_str = file.readline() ;  data_list_tp10.append(a_str.split())
    for j in range(89):  # skip 89 lines
       a_str = file.readline()
    a_str = file.readline() ;  data_list_tp100.append(a_str.split())
    for skip in range(200-4-89-8):  # Take samples 200 steps apart to give some independence
       a_str = file.readline()
    
file.close()

all_x_tm1   = np.array(data_list_tm1)
all_x_t     = np.array(data_list_t)
all_x_tp10  = np.array(data_list_tp10)
all_x_tp100 = np.array(data_list_tp100)

del(data_list_tm1)
del(data_list_t)
del(data_list_tp10)
del(data_list_tp100)

inputs_all_x_tm1 = np.zeros((K*n_run,8))
inputs_tm1       = np.zeros((K*n_run,4))
outputs_t        = np.zeros((K*n_run,1))
outputs_tp10     = np.zeros((K*n_run,1))
outputs_tp100    = np.zeros((K*n_run,1))
inputs_K_val     = np.zeros((K*n_run,1))

n_count = -1
for i in range(n_run):
    for j in range(8):
        n_count = n_count+1
        n1=(j-2)%8
        inputs_tm1[n_count,0] = all_x_tm1[i,n1]  
        n2=(j-1)%8
        inputs_tm1[n_count,1] = all_x_tm1[i,n2]        
        # i,j point itself
        inputs_tm1[n_count,2] = all_x_tm1[i,j]   
        n3=(j+1)%8
        inputs_tm1[n_count,3] = all_x_tm1[i,n3]
 
        outputs_t[n_count,0]     = all_x_t[i,j]    
        outputs_tp10[n_count,0]  = all_x_tp10[i,j]    
        outputs_tp100[n_count,0] = all_x_tp100[i,j]    

        inputs_all_x_tm1[n_count,:]=all_x_tm1[i,:]
        inputs_K_val[n_count,0] = int(j)

del(all_x_tm1)
del(all_x_t)
del(all_x_tp10)
del(all_x_tp100)

#Taken from D&B script...I presume this is a kind of 'normalisation'
max_train = 30.0
min_train = -20.0

inputs_all_x_tm1 = torch.FloatTensor(2.0*(inputs_all_x_tm1-min_train)/(max_train-min_train)-1.0)
inputs_tm1       = torch.FloatTensor(2.0*(inputs_tm1-min_train)/(max_train-min_train)-1.0)
outputs_t        = torch.FloatTensor(2.0*(outputs_t-min_train)/(max_train-min_train)-1.0)
outputs_tp10     = torch.FloatTensor(2.0*(outputs_tp10-min_train)/(max_train-min_train)-1.0)
outputs_tp100    = torch.FloatTensor(2.0*(outputs_tp100-min_train)/(max_train-min_train)-1.0)

print('inputs_all_x_tm1 shape : '+str(inputs_all_x_tm1.shape))
print('outputs_t shape ; '+str(outputs_t.shape))
no_samples=outputs_t.shape[0]
print('no samples ; ',+no_samples)


################################
print('Store data as Dataset') #
################################

class LorenzDataset(data.Dataset):
    """
    Lorenz Training dataset.
       
    Args:
        The arrays containing the training data
    """

    def __init__(self, inputs_all_x_tm1, inputs_tm1, outputs_t, outputs_tp10, outputs_tp100, inputs_K_val):

        self.inputs_all_x_tm1 = inputs_all_x_tm1
        self.inputs_tm1 = inputs_tm1
        self.outputs_t = outputs_t
        self.outputs_tp10 = outputs_tp10
        self.outputs_tp100 = outputs_tp100
        self.inputs_K_val = inputs_K_val

    def __getitem__(self, index):
	
        sample_tm1_all = inputs_all_x_tm1[index,:]
        sample_tm1 = inputs_tm1[index,:]
        sample_t = outputs_t[index]
        sample_tp10 = outputs_tp10[index,:]
        sample_tp100 = outputs_tp100[index,:]
        sample_inputs_K_val = inputs_K_val[index,:]

        return (sample_tm1, sample_t, sample_tp10, sample_tp100, sample_inputs_K_val, sample_tm1_all)

    def __len__(self):
        return outputs_t.shape[0]

# Instantiate the dataset
Lorenz_Dataset = LorenzDataset(inputs_all_x_tm1, inputs_tm1, outputs_t, outputs_tp10, outputs_tp100, inputs_K_val)

random_seed= 42
validation_split = .2

# Creating data indices for training and validation splits:
dataset_size = len(Lorenz_Dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
np.random.seed(random_seed)
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(Lorenz_Dataset, batch_size=hyper_params['batch_size'], num_workers=16, sampler=train_sampler)
val_loader   = torch.utils.data.DataLoader(Lorenz_Dataset, batch_size=hyper_params['batch_size'], num_workers=16, sampler=valid_sampler)


#######################################
print('')                             #
print('define AB 1st order iterator') #
print('')                             #
#######################################

def AB_1st_order_integrator(ref_state, h, n_steps):
    state = torch.tensor(np.zeros(ref_state.shape)).to(device).float()
    state[:,:] = ref_state[:,:]
    state_out = torch.tensor(np.zeros((n_steps,ref_state.shape[0],ref_state.shape[1])))
    out0 = torch.tensor(np.zeros((state.shape[0],8))).to(device).float()
    state_n = torch.tensor(np.zeros((state.shape[0],8,4)))
    for j in range(n_steps):  # iterate through time
        for k in range(8):    # iterate over each x point
            n1=(k-2)%8
            state_n[:,k,0] = state[:,n1]
            n2=(k-1)%8
            state_n[:,k,1] = state[:,n2]
            state_n[:,k,2] = state[:,k]
            n3=(k+1)%8
            state_n[:,k,3] = state[:,n3]
            out0[:,k] = h(state_n[:,k,:].to(device).float())[:,0]
        for k in range(8):
            state[:,k] = state[:,k] + out0[:,k]
        state_out[j,:,:] = (state[:,:])
    return(state_out)


#####################
print('Set up NNs') #
#####################

# Define matching sequential NNs, two hidden layers, with 20 neurons in each, and the tanh activation function

h_orig = nn.Sequential( nn.Linear( 4, 20), nn.Tanh(), 
                       nn.Linear(20, 20), nn.Tanh(), 
                       nn.Linear(20, 1 ) )
pickled_h = pickle.dumps(h_orig)

for model in ['1ts', '10ts_a.1', '10ts_a.3', '10ts_a1.',
              '100ts_a.1_b.1', '100ts_a.1_b.3','100ts_a.1_b1.',
              '100ts_a.3_b.1', '100ts_a.3_b.3','100ts_a.3_b1.',
              '100ts_a1._b.1', '100ts_a1._b.3','100ts_a1._b1.',]:

   h = pickle.loads(pickled_h)

   # parallelise and send to GPU
   if torch.cuda.device_count() > 1:
       print("Let's use", torch.cuda.device_count(), "GPUs!")
       h = nn.DataParallel(h)
   h.to(device)

   print('')
   print('######################')
   print('model is '+model)
   print('######################')

   opt = torch.optim.Adam(h.parameters(), lr=hyper_params['learning_rate'])   # Use adam optimiser for now, as simple to set up for first run

   with experiment.train():
      train_loss_batch = []
      train_loss_epoch = []
      val_loss_epoch = []
      for epoch in range(hyper_params['num_epochs']):
         print('Epoch {}/{}'.format(epoch, hyper_params['num_epochs'] - 1))
         print('-' * 10)
         train_loss_temp = 0.0
         val_loss_temp = 0.0
      
         for tm1, t, tp10, tp100, K, tm1_all in train_loader:
            tm1_all = tm1_all.to(device).float()
            tm1     = tm1.to(device).float()
            t       = t.to(device).float()
            tp10    = tp10.to(device).float()
            tp100   = tp100.to(device).float()
            K = K.to(device).long()
            h.train(True)
            estimate1 = tm1[:,2,None] + h(tm1[:,:])
            if model == '1ts':
               loss = (estimate1 - t).abs().mean()  # mean absolute error
            elif model[0:4] == '10ts':
               iterations = AB_1st_order_integrator(tm1_all[:,:], h, 10)
               estimate10_temp = iterations[9,:,:]
               estimate10 = estimate10_temp[range(estimate10_temp.shape[0]), K.flatten()].reshape(-1,1)
               alpha = float(model[6:8])
               loss = ( (estimate1.float().to(device) - t).abs() + alpha*(estimate10.float().to(device) - tp10).abs() ).mean()
            elif model[0:5] == '100ts':
               iterations = AB_1st_order_integrator(tm1_all[:,:], h, 100)
               estimate10_temp = iterations[9,:,:]
               estimate10 = estimate10_temp[range(estimate10_temp.shape[0]), K.flatten()].reshape(-1,1)
               estimate100_temp = iterations[99,:,:]
               estimate100 = estimate100_temp[range(estimate100_temp.shape[0]), K.flatten()].reshape(-1,1)
               alpha = float(model[7:9])
               beta = float(model[11:13])
               loss = ( (estimate1 - t).abs() + alpha*(estimate10.float().to(device) - tp10).abs() + beta*(estimate100.float().to(device) - tp100).abs() ).mean()
            else:
               print('ERROR!! model not known!')
            loss.backward()
            opt.step()
            opt.zero_grad()
            train_loss_batch.append(loss.item())
            train_loss_temp += loss.item()
            # log to comet.ml
            experiment.log_metric('rms', loss.item())
       
   with experiment.test():
      for tm1, t, tp10, tp100, K, tm1_all in val_loader:
         tm1_all = tm1_all.to(device).float()
         tm1     = tm1.to(device).float()
         t       = t.to(device).float()
         tp10    = tp10.to(device).float()
         tp100   = tp100.to(device).float()
         K = K.to(device).long()
         h.train(False)
         estimate1 = tm1[:,2,None] + h(tm1[:,:])
         if model == '1ts':
            loss = (estimate1 - t).abs().mean()  # mean absolute error
         elif model[0:4] == '10ts':
            iterations = AB_1st_order_integrator(tm1_all[:,:], h, 10)
            estimate10_temp = iterations[9,:,:]
            estimate10 = estimate10_temp[range(estimate10_temp.shape[0]), K.flatten()].reshape(-1,1)
            alpha = float(model[6:8])
            loss = ( (estimate1.float().to(device) - t).abs() + alpha*(estimate10.float().to(device) - tp10).abs() ).mean()
         elif model[0:5] == '100ts':
            iterations = AB_1st_order_integrator(tm1_all[:,:], h, 100)
            estimate10_temp = iterations[9,:,:]
            estimate10 = estimate10_temp[range(estimate10_temp.shape[0]), K.flatten()].reshape(-1,1)
            estimate100_temp = iterations[99,:,:]
            estimate100 = estimate100_temp[range(estimate100_temp.shape[0]), K.flatten()].reshape(-1,1)
            alpha = float(model[7:9])
            beta = float(model[11:13])
            loss = ( (estimate1 - t).abs() + alpha*(estimate10.float().to(device) - tp10).abs() + beta*(estimate100.float().to(device) - tp100).abs() ).mean()
         else:
            print('ERROR!! model not known!')
         val_loss_temp += loss.item()
         experiment.log_metric('rms', loss.item())

      train_loss_epoch.append(train_loss_temp / len(train_indices))
      val_loss_epoch.append(val_loss_temp / len(val_indices))
      print('Training Loss: {:.8f}'.format(train_loss_epoch[-1]))
      print('Validation Loss: {:.8f}'.format(val_loss_epoch[-1]))
   
   fig = plt.figure()
   ax1 = fig.add_subplot(211)
   ax1.plot(train_loss_batch)
   ax1.set_xlabel('Batches')
   ax1.set_ylabel('Loss')
   ax1.set_yscale('log')
   ax2 = fig.add_subplot(212)
   ax2.plot(train_loss_epoch)
   ax2.plot(val_loss_epoch)
   ax2.legend(['Training loss','Validation loss'])
   ax2.set_xlabel('Epochs')
   ax2.set_ylabel('Loss')
   ax2.set_yscale('log')
   plt.subplots_adjust(hspace=0.4, top=0.9, bottom=0.12, left=0.08, right=0.85)
   plt.savefig('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/PLOTS/trainingloss_'+model+'_'+str(n_run)+'.png',  bbox_inches = 'tight', pad_inches = 0.1)
   
   torch.save({'h_state_dict': h.state_dict(),
               'opt_state_dict': opt.state_dict(),
   	   }, '/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/MODELS/'+model+'_model_'+str(n_run)+'.pt')
