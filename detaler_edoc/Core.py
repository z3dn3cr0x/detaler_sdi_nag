#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import time
import requests
from os import path,scandir
import datetime
import dpkt
from tqdm import tqdm
import tarfile
import logging
import pickle
from tensorboardX import SummaryWriter
import pandas as pd

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size,output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.elu(self.map1(x))
        x = F.sigmoid(self.map2(x))
        return F.sigmoid(self.map3(x))
    
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.elu(self.map1(x))
        x = F.elu(self.map2(x))
        return F.sigmoid(self.map3(x))
    
    # Data params
    
    # Model params
g_hidden_size = 50 # Generator complexity

d_hidden_size = 50 # Discriminator complexity
d_output_size = 1 # Single dimension for 'real' vs. 'fake'

minibatch_size = 100

d_learning_rate = 2e-4 # 2e-4
g_learning_rate = 2e-4
optim_betas = (0.9, 0.999)
num_epochs = 2
print_interval = 200
d_steps = 5 # 'k' steps in the original GAN paper. Can put the discriminator on higher training freq than generator

g_steps = 1

new_number_data_bytes = False
num_databytes=None
num_packets_limit=None
pcap_path="./data/training.pcap"

torch.set_default_tensor_type('torch.cuda.FloatTensor')

bitstream, packet_size = data_to_bitstream(num_packets_limit=num_packets_limit, num_databytes=num_databytes, new_number_data_bytes=new_number_data_bytes, pcap_file=pcap_path)

print(packet_size)

create_attack_packets(packet_size)

g_input_size = packet_size


data_class = Data_preporation(data=bitstream, batch_size=minibatch_size)

data_class.reset_batch_pointer()

gi_sampler = get_generator_input_sampler()

G = Generator(input_size=packet_size, hidden_size=g_hidden_size, output_size=packet_size)
G.cuda()

D = Discriminator(input_size=packet_size, hidden_size=d_hidden_size, output_size=d_output_size)
D.cuda()

print(G)

print(D)

criterion = nn.BCELoss() # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss

d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate, betas=optim_betas)

g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate, betas=optim_betas)

timestr = time.strftime("%d%m%Y-%H%M")

writer = SummaryWriter("C:\\tensorboard\\pytorch\\%s"%timestr)

test_writer = SummaryWriter("C:\\tensorboard\\pytorch\\%s_test"%timestr)

for epoch in range(num_epochs):
    data_class.reset_batch_pointer()
    attacks = create_attack_packets(packet_size)
    seed=len(attacks)//4
    print(seed)
    
    testing,validate = attacks[:seed*3],attacks[seed:]
    new_attack_data = process_attacks(packet_size)

    for batches in range(data_class.num_batches):
        tick = (epoch*data_class.get_num_batches()) + batches
        
        d_real_data,test = data_class.next_batch()
        d_real_data=torch.from_numpy(d_real_data)
        d_real_data =d_real_data.float() 
        
        #test =)
    
        d_real_data_tensor = Variable(d_real_data.cuda())
        
        #print(d_real_data_tensor)
        # print(x)
        for d_index in range(d_steps):
            # 1. Train D on real+fake
            D.zero_grad()
        
            # 1A: Train D on real
        
            d_real_decision = D(d_real_data_tensor)
            # print(d_real_decision)
            d_real_error = criterion(d_real_decision, Variable(torch.ones(100).cuda())) 
            # ones = true
        
            # print(d_real_error)
            d_real_error.backward() # compute/store gradients, but don't change params
        
            # 1B: Train D on fake
            d_gen_input = Variable(torch.rand(minibatch_size, g_input_size))
            d_fake_data = G(d_gen_input).detach() # detach to avoid training G on these labels
            d_fake_decision = D(d_fake_data)
            d_fake_error = criterion(d_fake_decision, Variable(torch.zeros(100).cuda())) # zeros = fake
            d_fake_error.backward()
            # d_optimizer.step()
            d_optimizer.step()
        
        for g_index in range(g_steps):
            # 2. Train G on D's response (but DO NOT train D on these labels)
            G.zero_grad()
            
            gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
            g_fake_data = G(gen_input)
            dg_fake_decision = D(g_fake_data)
            
            g_error = criterion(dg_fake_decision, Variable(torch.ones(100).cuda()))
            # we want to fool, so pretend it's all genuine
            
            g_error.backward()
            g_optimizer.step() # Only optimizes G's parameters
            
            writer.add_scalar('loss/d_real', d_real_error, tick)
            
            writer.add_scalar('loss/d_fake', d_fake_error, tick)
            
            writer.add_scalar('loss/g', g_error, tick)
            
        if tick % 10 == 0:
            d_attack_data = torch.from_numpy(testing)
            d_attack_data = d_attack_data.float()
            d_attack_test = Variable(d_attack_data.cuda())
            d_attack = D(d_attack_test)
            
            running_attack_ac = metric(d_attack, zeroes=True)
            
            d_attack_error = criterion(d_attack, Variable(torch.zeros_like(d_attack.data)))
            # print(running_attack_ac)
            d_attack_error.backward()
            d_optimizer.step() # Only optimizes D's
            
        if tick % 10 == 0:
            print(tick)
            
            running_fake_ac = metric(d_fake_decision,zeroes=True)
            running_real_ac=metric(d_real_decision,zeroes=False)
            
            writer.add_scalar('Prediction/real', running_real_ac, tick)
            
            writer.add_scalar('Prediction/fake', running_fake_ac, tick)
            
            for name, param in D.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), tick)
                
                d_real_data = torch.from_numpy(test)
                d_real_data = d_real_data.float()
                d_real_test = Variable(d_real_data.cuda())
                d_test = D(d_real_test).detach()
                running_test_ac = metric(d_test,zeroes=False)
                
                writer.add_graph(D, d_real_test)
                
                d_gen_input = Variable(torch.rand(minibatch_size, g_input_size))
                
                d_fake_data = G(d_gen_input).detach() # detach to avoid training G on these labels
                
                d_test_fake_decision = D(d_fake_data)
                
                running_fake_test_ac = metric(d_test_fake_decision,zeroes=True)
                
                writer.add_graph(G, d_gen_input)
                
                test_writer.add_scalar('Prediction/Test/real', running_test_ac, tick)
                
                test_writer.add_scalar('Prediction/Test/fake', running_fake_test_ac, tick)
                
                d_attack_data = torch.from_numpy(new_attack_data)
                d_attack_data = d_attack_data.float()
                d_attack_test = Variable(d_attack_data.cuda())
                d_attack = D(d_attack_test).detach()
                
                running_attack_ac = metric(d_attack,zeroes=True)
                
                test_writer.add_scalar('Prediction/attacks', running_attack_ac, tick)
                
                print("%s/%s: D: %s/%s G: %s (Real: %s, Fake: %s), Test Real: %s, Test Fake: %s " % (574 batches, data_class.num_batches, d_fake_error.item(),                d_real_error.item(), g_error.item(), running_real_ac, running_fake_ac, running_test_ac, running_fake_test_ac))
                
                print("Real attack from dataset:",running_attack_ac)
                
    new_attack_data_numpy = torch.from_numpy(new_attack_data)
    new_attack_data_numpy = new_attack_data_numpy.float()
    new_attack_variable = Variable(new_attack_data_numpy.cuda())
    new_attack_prediction = D(new_attack_variable).detach()
                
    running_bob = metric(new_attack_prediction,zeroes=True)
    
    print(new_attack_prediction.data)
    print(running_bob,"Of DARPA Week 4 & 5 Real Attack Detected!",)
    
    if epoch % 1 == 0:
        gen_input = Variable(gi_sampler(10000, g_input_size).cuda())
        
        gen_sample = G(gen_input).detach()
        print(gen_sample)
        
        gen_sample=gen_sample.cpu().data.numpy()
        gen_sample=np.around(gen_sample)
        print(gen_sample)
        
        sample_bytes = sample_to_bytes(gen_sample)
        
        # print()
        write_bytes_to_pcap(list_of_bytes=sample_bytes, file_path="./data/gen/gan_packets.pcap")

