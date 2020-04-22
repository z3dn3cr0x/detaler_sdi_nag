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

##### Pre-prossesing:

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
    
def data_to_bitstream(num_packets_limit=None,num_databytes=None,new_number_data_bytes=False, pcap_file="./data/training.pcap"):
    if num_databytes is None:
        packet_size = None
    else:
        packet_size = 304 + (num_databytes*8)

    if path.isfile("./data/preprocessed_bits.pkl") and new_number_data_bytes==False:
        logging.info("Using saved processed data...")
        trimmed_bits_numpy = np.load(open("./data/preprocessed_bits.pkl","rb"))
        return trimmed_bits_numpy,trimmed_bits_numpy.shape[1]
    elif (path.isfile("./data/untrimmed_bytes.pkl")):
        logging.info("Using saved untrimmed processed data...")
        untrimmed_bytes_list = np.load(open("./data/untrimmed_bytes.pkl","rb"))
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
                    
        pickle.dump(untrimmed_bytes_list, open("./data/untrimmed_bytes.pkl", "wb"), protocol=4)
        
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
    
    # print(trimmed_bits_list[:2])
    assert len(trimmed_bits_list) == len(untrimmed_bytes_list), "Trimmed bits must be the same len as untrimmed, "         "found %s > %s" % (len(trimmed_bits_list), len(untrimmed_bytes_list),)
        
    #print(len(max(trimmed_bits_list, key=len)))
    #print(len(max(trimmed_bits_list, key=len)))
    trimmed_bits_numpy = np.asanyarray([list(map(int,string)) for lists in tqdm(trimmed_bits_list) for string in lists])
        
    print(trimmed_bits_numpy)

    assert trimmed_bits_numpy.shape == (num_packets_limit,packet_size) 
        or num_packets_limit == None, ""         "Trimmed data dont match origanal data, found %s > %s" % (trimmed_bits_numpy.shape, (num_packets_limit,packet_size)) 
        #trimmed_bits_numpy.dump("./data/preprocessed_bits.pkl")
        pickle.dump(trimmed_bits_numpy,open("./data/preprocessed_bits.pkl","wb"),protocol=4)
            
    return trimmed_bits_numpy,trimmed_bits_numpy.shape[1]
    
def get_generator_input_sampler():
    return lambda m, n: torch.rand(m, n)
                      
                                                                                                  
def create_attack_packets(packet_size):
    if path.isfile("./data/preprocessed_attacks.pkl"):
        logging.info("Using saved processed data...")
        trimmed_bits_numpy = np.load(open("./data/preprocessed_attacks.pkl", "rb"))
        return trimmed_bits_numpy
    else:
        pcap = dpkt.pcap.Reader(open("./data/inside.pcap",'rb'))
        attack_list = ["080101"]
        
        df = pd.DataFrame(pd.read_csv(open("./data/attacks.csv", "r"), sep=";", encoding="UTF-8"))
        attack_list.extend(["".join(x[1].split(":")) for x in df["Start_Time"].iteritems()])
        
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
        pickle.dump(trimmed_bits_numpy, open("./data/preprocessed_attacks.pkl", "wb"), protocol=4)

        return trimmed_bits_numpy

    #print('Timestamp: ',)
    
def process_attacks(packet_size):
    if path.isfile("./data/week_4.pkl"):
        # logging.info("Using saved processed data...")
        trimmed_bits_numpy = np.load(open("./data/week_4.pkl", "rb"))
        return trimmed_bits_numpy
    else:
        df = pd.DataFrame(pd.read_csv(open("./data/week_4.csv", "r",encoding="utf-8"), sep=";", encoding="UTF-8"))
        #print(df.head())
        #print(df["id"])
        match_dict={}
        for entry in scandir("./data/testing"):
            attack_list=[]
            
            if entry.is_file() and entry.name is not None:
                #pcap = dpkt.pcap.Reader(open("./data/week_4.pcap", 'rb'))
                name= entry.name[:2]
                #print()
                test =[tuple(y.split(".")) for x,y in df["id"].iteritems()]
                # print(test)
                for x in test:
                    # print(x[0],name)
                    if x[0] == name:
                        print("match")
                        
                        # print(x[1])
                        attack_list.append(x[1])
                        
                        #print(attack_list)
            match_dict[entry.name[:2]]=attack_list
            
            attack_packets = []
            
            for k,v in match_dict.items():
                pcap = dpkt.pcap.Reader(open("./data/testing/%s.pcap"%k,'rb'))
                
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
                        
                        print(time_long_string[:6], attack_list)
                        
                        if time_long_string[:6] in attack_list:
                            bit_string = ""
                            print("Found")
                            # attack_list.pop()
                            print(attack_list)
                            for bytes in list(buf):
                                #print(bytes)
                                bit_string += '{0:08b}'.format(bytes)
                                
                                #print(bit_string)
                                #break
                                attack_packets.append(bit_string)
                                attack_list.pop(0)
                                
        trimmed_bits_list = [sublist[:packet_size] for ublist in attack_packets]
        
        # print(trimmed_bits_list)
        trimmed_bits_list = [[(x + "0" * (packet_size - len(x)))] for x in tqdm(trimmed_bits_list)]
        
        # print(trimmed_bits_list[:2])
        trimmed_bits_numpy = np.asanyarray([list(map(int, string)) for lists in tqdm(trimmed_bits_list) for string in lists])
        
        # print(trimmed_bits_numpy)
        pickle.dump(trimmed_bits_numpy, open("./data/week_4.pkl", "wb"), protocol=4)
        
        return trimmed_bits_numpy
    
def write_bytes_to_pcap(list_of_bytes,file_path="./data/gan_packets.pcap"):
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










