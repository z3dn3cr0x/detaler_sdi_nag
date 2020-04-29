#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import lib
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

def get_generator_input_sampler():
    return lambda m, n: torch.rand(m, n)

def write_bytes_to_pcap(list_of_bytes,file_path="F:\\tdata\\Data\\gan_packets.pcap"):
    logging.info("Writing %s packets to file"%len(list_of_bytes))
    
    with open(file_path, "wb") as f:
        fd= dpkt.pcap.Writer(f)
        for packet in tqdm(list_of_bytes):
            #print(packet)
            fd.writepkt(packet, time.time())
            #f.flush()

def metric(d_attack,zeroes):
    if zeroes:
        d_attack = np.around(np.sum(np.equal((np.around(d_attack.cpu().data.numpy())),np.zeros_like(d_attack.cpu().data.numpy()))
                    / len(d_attack.cpu().data.numpy())),6)
    else:
        d_attack = np.around(np.sum(np.equal((np.around(d_attack.cpu().data.numpy())),np.ones_like(d_attack.cpu().data.numpy()))
                    / len(d_attack.cpu().data.numpy())),)
    return d_attack


# create logger
def logging_start():
# create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(level=logging.DEBUG)
    
# create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

# create formatter
    formatter = logging.Formatter('%(asctime)s: %(levelname)s: %(message)s')

# add formatter to ch
    ch.setFormatter(formatter)
                                  
# add ch to logger
    logger.addHandler(ch)
    
    return logger

logger = logging_start()

class Data_preporation():
    def __init__(self,data,batch_size):
        self.data=data
        self.batch_size=batch_size
        self.pointer = 0

        self.data=self.data[:-1000]
        self.test_data=self.data[-1000:]
        self.batch_creation()

    def batch_creation(self):
        self.num_batches = self.data.shape[0] // self.batch_size
        #print(self.num_batches)
        self.x_train = self.data[:self.num_batches * self.batch_size]
        self.x_batches = np.split(self.x_train, self.num_batches)
        #print(x_train.shape)
        #print(x_batches)
        
    def next_batch(self):
        #x,y = db.next_batch()
        self.x_batch = self.x_batches[self.pointer]
        self.pointer += 1
        # #x_test, y_test = self.x_test_batches[self.pointer], self.y_test_batches[self.pointer]
        # return x, y, self.x_test, self.y_test
        return self.x_batch,self.test_data
    
    def reset_batch_pointer(self):
        self.pointer = 0
        
    def get_num_batches(self):
        return self.num_batches
    
def data_to_bitstream(num_packets_limit=None,num_databytes=None,new_number_data_bytes=False, pcap_file="F:\\tdata\\Data\\training.pcap"):
    if num_databytes is None:
        packet_size = None
    else:
        packet_size = 304 + (num_databytes*8)

    if path.isfile("F:\\tdata\\Data\\preprocessed_bits.pkl") and new_number_data_bytes==False:
        logging.info("Using saved processed data...")
        trimmed_bits_numpy = np.load(open("F:\\tdata\\Data\\preprocessed_bits.pkl","rb"),allow_pickle=True)
        return trimmed_bits_numpy,trimmed_bits_numpy.shape[1]
    elif (path.isfile("F:\\tdata\\Data\\untrimmed_bytes.pkl")):
        logging.info("Using saved untrimmed processed data...")
        untrimmed_bytes_list = np.load(open("F:\\tdata\\Data\\untrimmed_bytes.pkl","rb"),allow_pickle=True)
    else: 
        logger.info("Reading Packets from pcap file %s" % pcap_file)
        pcap = dpkt.pcap.Reader(open(pcap_file, 'rb'))
        TCP_packets = []
        count = 0
            
            
        untrimmed_bytes_list=[]
        for ts, buf in tqdm(pcap):
            bit_string=""
            # print(type(buf))
            for bytes in list(buf):
                bit_string += '{0:08b}'.format(bytes)
                
            #print("packet size",len(bit_string))
            untrimmed_bytes_list.append(bit_string)
            count+=1

            if count == num_packets_limit and num_packets_limit is not None:
                break
                    
        pickle.dump(untrimmed_bytes_list, open("F:\\tdata\\Data\\untrimmed_bytes.pkl", "wb"), protocol=4)
        
    max_bytes = int(len(max(untrimmed_bytes_list,key=len)))
        
    if packet_size is None:
        packet_size=max_bytes
            
    #print(packet_size > max_bytes)
    #print()
    assert packet_size <= max_bytes, "The packets size cant extend maximum bytes, found %s > %s"% (packet_size,max_bytes,)
        
    print("\rLimit on Number of Packets: {}, Limit on Data Bytes: {}, Packet Size:{}".format(num_packets_limit, num_databytes,packet_size))
    
    #logging.info()
    trimmed_bits_list=[sublist[:packet_size] for sublist in untrimmed_bytes_list]
        
    #print(trimmed_bits_list[:2])
    #print(len(max(trimmed_bits_list, key=len)))
    #print(len(trimmed_bits_list[0]))
    #print((untrimmed_bytes_list[0]))
        
    trimmed_bits_list = [[(x + "0" * (packet_size - len(x)))] for x in tqdm(trimmed_bits_list)]
    #print(trimmed_bits_list)
    
    #print(trimmed_bits_list[:2])
    assert len(trimmed_bits_list) == len(untrimmed_bytes_list), "Trimmed bits must be the same len as untrimmed, "         "found %s > %s" % (len(trimmed_bits_list), len(untrimmed_bytes_list),)
        
    #print(len(max(trimmed_bits_list, key=len)))
    #print(len(max(trimmed_bits_list, key=len)))
    trimmed_bits_numpy = np.asanyarray([list(map(int,string)) for lists in tqdm(trimmed_bits_list) for string in lists])
        
    # print(trimmed_bits_numpy)

    assert trimmed_bits_numpy.shape == (num_packets_limit,packet_size) or num_packets_limit == None, "        Trimmed data dont match origanal data, found %s > %s" % (trimmed_bits_numpy.shape, (num_packets_limit,packet_size)) 
        #trimmed_bits_numpy.dump("./data/preprocessed_bits.pkl")
    pickle.dump(trimmed_bits_numpy,open("F:\\tdata\\Data\\preprocessed_bits.pkl","wb"),protocol=4)
    
    #print the first 100 values of first packet
    #print("The first 100 values of first packet: ",trimmed_bits_numpy[0:1,0:100])
    return trimmed_bits_numpy,trimmed_bits_numpy.shape[1]

def create_attack_packets(packet_size):
    if path.isfile("F:\\tdata\\Data\\preprocessed_attacks.pkl"):
        logging.info("Using saved processed data...")
        trimmed_bits_numpy = np.load(open("F:\\tdata\\Data\\preprocessed_attacks.pkl", "rb"), allow_pickle=True)
        return trimmed_bits_numpy
    else:
        pcap = dpkt.pcap.Reader(open("F:\\tdata\\Data\\inside.pcap",'rb'))
        attack_list = []
        
        df = pd.DataFrame(pd.read_csv(open("F:\\tdata\\Data\\attacks.csv",'r'), sep=";", encoding = 'utf-8'))
        attack_list.extend(["".join(x[1].split(":")) for x in df["StartTime"].iteritems()])
        
        print(attack_list)
        print(len(attack_list))
        
        untrimmed_bytes_list = []
        count = 0

        attack_packets = []
        
        for ts, buf in tqdm(pcap):
            ts = str(datetime.datetime.utcfromtimestamp(ts) - datetime.timedelta(hours=5))
            
            date, time = str(ts)[:10], str(ts)[11:]
            # print(date,time)

            time_long_string = "".join(time.split(":"))

            # print(time_long_string[:6],attack_list)
            if time_long_string[:6] in attack_list:
                bit_string = ""
                
                # print("Found attack")
                # eth=dpkt.ethernet.Ethernet(buf)
                # print(repr(eth))
                for bytes in list(buf):
                    # print(bytes)
                    bit_string += '{0:08b}'.format(bytes)
                    
                    # print(bit_string)
                    # break
                attack_packets.append(bit_string)
                # print(attack_packets)
                # print("%s Attacks found" % count)
            count += 1
            
            if time_long_string[:6] == attack_list[-1]:
                break
                
                # if count >= 10000:
                # break
        trimmed_bits_list = [sublist[:packet_size] for sublist in attack_packets]
        
        # print(trimmed_bits_list)
        trimmed_bits_list = [[(x + "0" * (packet_size - len(x)))] for x in tqdm(trimmed_bits_list)]
        
        # print(trimmed_bits_list[:2])
        trimmed_bits_numpy = np.asanyarray([list(map(int, string)) for lists in tqdm(trimmed_bits_list) for string in lists])
        
        # print(trimmed_bits_numpy)
        pickle.dump(trimmed_bits_numpy, open("F:\\tdata\\Data\\preprocessed_attacks.pkl", "wb"), protocol=4)
        
#         print("The first 100 values of first packet: ",trimmed_bits_numpy[0:1,0:100])
        
        return trimmed_bits_numpy
    
def process_attacks(packet_size):
    if path.isfile("F:\\tdata\\Data\\atk4.pkl"):
        # logging.info("Using saved processed data...")
        trimmed_bits_numpy = np.load(open("F:\\tdata\\Data\\atk4.pkl", "rb"), allow_pickle=True)
        return trimmed_bits_numpy
    else:
        df = pd.DataFrame(pd.read_csv(open("F:\\tdata\\Data\\atk_w4.csv", "r",encoding="utf-8"), sep=";", encoding="UTF-8"))
        #print(df.head())
        #print(df["id"])
        match_dict={}
        for entry in scandir("F:\\tdata\\Data\\atkpcap"):
            attack_list=[]
           
            if entry.is_file() and entry.name is not None:
                #pcap = dpkt.pcap.Reader(open("./data/week_4.pcap", 'rb'))
                name= entry.name[:2]
                #print()
                test =[tuple(str(y).split(".")) for x,y in df["IDnum"].iteritems()]
                # print(test)
                for x in test:
                    # print(x[0],name)
                    if x[0] == name:
                        #print("match")
                       
                        # print(x[1])
                        attack_list.append(x[1])
                       
                        #print(attack_list)
            match_dict[entry.name[:2]]=attack_list
           
        attack_packets = []
        for k,v in match_dict.items():
                pcap = dpkt.pcap.Reader(open("F:\\tdata\\Data\\atkpcap\\%s.pcap"%k,'rb'))
               
                attack_list = v
                #print(v)
                untrimmed_bytes_list = []
                count = 0
               
                # for ts, buf in tqdm(pcap):
                for ts, buf in tqdm(pcap):
                   
                    if not attack_list:
                        break
                   
                    else:
                        ts = str(datetime.datetime.utcfromtimestamp(ts) - datetime.timedelta(hours=5))
                       
                        date, time = str(ts)[:10], str(ts)[11:]
                       
                        # print(date,time)
                        time_long_string = "".join(time.split(":"))
                       
                        #print(time_long_string[:6], attack_list)
                       
                        if time_long_string[:6] in attack_list:
                            bit_string = ""
                            print("Found")
                            # attack_list.pop()
                            #print(attack_list)
                            for bytes in list(buf):
                                #print(bytes)
                                bit_string += '{0:08b}'.format(bytes)
                               
                                #print(bit_string)
                                #break
                                attack_packets.append(bit_string)
                                attack_list.pop(0) if attack_list else None    
               
                               
        trimmed_bits_list = [sublist[:packet_size] for sublist in attack_packets]
       
        # print(trimmed_bits_list)
        trimmed_bits_list = [[(x + "0" * (packet_size - len(x)))] for x in tqdm(trimmed_bits_list)]
       
        # print(trimmed_bits_list[:2])
        trimmed_bits_numpy = np.asanyarray([list(map(int, string)) for lists in tqdm(trimmed_bits_list) for string in lists])
       
        # print(trimmed_bits_numpy)
        pickle.dump(trimmed_bits_numpy, open("F:\\tdata\\Data\\atk4.pkl", "wb"), protocol=4)
       
        return trimmed_bits_numpy


# In[2]:


def sample_to_bytes(sample_array):
    bytes_list=[]
    logger.info(("Formatting {} samples of {} bits to bytes".format(sample_array.shape[0],sample_array.shape[1])))
    print(("Formatting {} samples of {} bits to bytes".format(sample_array.shape[0],sample_array.shape[1])))

    sample_list = sample_array.astype(int).tolist()
    #print("sample_list",sample_list)
    bit_strings = ["".join(map(str, lists)) for lists in tqdm(sample_list)]
    #print(bit_strings)
    for x in tqdm(bit_strings):
        bytes_list.append(bytes([int(x[i:i+8],2) for i in range(0,len(x),8)]))
    logging.debug(bytes_list[:2])
    print(len(bytes_list[1]))
    return bytes_list


# In[3]:


class Generator(nn.Module):
    def __init__(self, input_size, hidden_size,output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.elu(self.map1(x))
        x = torch.sigmoid(self.map2(x))
        return torch.sigmoid(self.map3(x))
    
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        #print("x.size: ")
        #print(x.size())
        x = F.elu(self.map1(x))
        x = F.elu(self.map2(x))
        #print("x.size now: ")
        #print(x.size())
        return torch.sigmoid(self.map3(x))


# In[4]:


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
pcap_path="F:\\tdata\\Data\\training.pcap"

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
print("===============================")
#print(d_optimizer)
g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate, betas=optim_betas)
print("===============================")
#print(g_optimizer)
timestr = time.strftime("%d%m%Y-%H%M")

writer = SummaryWriter("C:\\tensorboard\\pytorch\\%s"%timestr)

test_writer = SummaryWriter("C:\\tensorboard\\pytorch\\%s_test"%timestr)

print(num_epochs)


# In[5]:


for epoch in range(num_epochs):
    print("1. Đây là epoch: ")
    print(epoch)
    data_class.reset_batch_pointer()
    attacks = create_attack_packets(packet_size)
    print("Len: ")
    print(len(attacks))
    seed=len(attacks)//4
    print("===========================================")
    print("2. Đây là seed: ")
    print(seed)
    
    testing,validate = attacks[:seed*3],attacks[seed:]
    print("===========================================")
    print("3.0 đây là testing: ")
    print(testing)
    print("=============")
    print("3.1 sau testing là validate")
    print(validate)

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
            #line 78 --> (testing) --> np.asanyarray(testing)
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
                # comment tạm
                #writer.add_graph(D, d_real_test)
                
                d_gen_input = Variable(torch.rand(minibatch_size, g_input_size))
                
                d_fake_data = G(d_gen_input).detach() # detach to avoid training G on these labels
                
                d_test_fake_decision = D(d_fake_data)
                
                running_fake_test_ac = metric(d_test_fake_decision,zeroes=True)
                # comment tạm
                #writer.add_graph(G, d_gen_input)
                
                test_writer.add_scalar('Prediction/Test/real', running_test_ac, tick)
                
                test_writer.add_scalar('Prediction/Test/fake', running_fake_test_ac, tick)
                
                d_attack_data = torch.from_numpy(new_attack_data)
                d_attack_data = d_attack_data.float()
                d_attack_test = Variable(d_attack_data.cuda())
                d_attack = D(d_attack_test).detach()
                
                running_attack_ac = metric(d_attack,zeroes=True)
                
                test_writer.add_scalar('Prediction/attacks', running_attack_ac, tick)
                # comment tạm
                #print("%s/%s: D: %s/%s G: %s (Real: %s, Fake: %s), Test Real: %s, Test Fake: %s " % (data_class.num_batches, d_fake_error.item(), d_real_error.item(), g_error.item(), running_real_ac, running_fake_ac, running_test_ac, running_fake_test_ac))
                
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
        write_bytes_to_pcap(list_of_bytes=sample_bytes, file_path="F:\\tdata\\Data\\gen\\gan_packets.pcap")


# In[ ]:




