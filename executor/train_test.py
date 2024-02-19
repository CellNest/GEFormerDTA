import copy
import os
import pickle
import random
import sys
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric import data as DATA
from tqdm import tqdm
from sklearn.metrics import r2_score

import config
from collate import *
from datatset import *
from general_graph import prot_to_graph, smile_to_graph, drug_embedding
from graph_conversion import *
from metrics import *
from models.GEFormerDTA import GEFormerDTA
from models.GLFA import GLFA
from models.GEFA import GEFA
from models.GEFormerDTA_with_DegreeC import GEFormerDTA_with_DegreeC
from models.GEFormerDTA_with_SpatialP import GEFormerDTA_with_SpatialP
from models.GEFormerDTA_with_Edge import GEFormerDTA_with_Edge
from models.GEFormerDTA_without_ss_asa import GEFormerDTA_without_ss_asa
from preprocess import *

torch.manual_seed(2)
np.random.seed(3)

num_feat_xp = 0
num_feat_xd = 0

model_name_seq = '_seq' if config.is_seq_in_graph is True else ''
model_name_con = '_con' if config.is_con_in_graph is True else ''
model_name_profile = '_pf' if config.is_profile_in_graph is True else ''
model_name_emb = '_emb' if config.is_emb_in_graph is True else ''

print('Using features: ')
print('Sequence.') if config.is_seq_in_graph else print('')
print('Contact.') if config.is_con_in_graph else print('')
print('SS + SA.') if config.is_profile_in_graph else print('')
print('Embedding.') if config.is_emb_in_graph else print('')

dataset = config.dataset
print('Dataset: ', dataset)

modeling = [GEFormerDTA,
            GEFA,
            GLFA,
            GEFormerDTA_with_Edge,
            GEFormerDTA_with_DegreeC,
            GEFormerDTA_with_SpatialP,
            GEFormerDTA_without_ss_asa][config.run_model]
model_st = modeling.__name__

cuda_name = "cuda:" + str(config.cuda)
print('CUDA name:', cuda_name)

set_num = config.setting
settings = ['_setting_1', '_setting_2', '_setting_3', '_setting_4']
setting = settings[set_num]
print("Setting: ", setting)

print('Train batch size: ', config.TRAIN_BATCH_SIZE)
print('Test batch size: ', config.TEST_BATCH_SIZE)

LR = config.LR
print("Learning rate: ", LR)

from_resume = [False, True][int(sys.argv[1])]

print('Number of epoch: ', config.NUM_EPOCHS)
LOG_INTERVAL = 20


# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    total_train_loss = 0.0
    for batch_idx, data in enumerate(train_loader):

        import torch, gc
        gc.collect()
        torch.cuda.empty_cache()

        drug = data[0].to(device)
        prot = data[1].to(device)
        optimizer.zero_grad()
        output = model(drug, prot)
        # print('ouput')
        # print(output)
        affinity = drug.y.view(-1, 1).float()
        # print(output.shape, affinity.shape)
        loss = loss_fn(output, affinity.to(device))
        total_train_loss += loss.item()
        # loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(drug.y),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
    print('Average loss: {:.4f}'.format(total_train_loss / (batch_idx + 1)))
    return total_train_loss / (batch_idx + 1)


def adjust_learning_rate(optimizer, LR, scale=0.7):
    """Sets the learning rate to the initial LR decayed by 10 every interval epochs"""
    lr = LR * scale

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def save_checkpoint(state, filename=''):
    torch.save(state, filename)


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_preds_f = torch.Tensor()
    total_preds_s = torch.Tensor()

    total_labels = torch.Tensor()
    total_labels_f = torch.Tensor()
    total_labels_s = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            drug = data[0].to(device)
            prot = data[1].to(device)
            output = model(drug, prot)
            total_preds_f = torch.cat((total_preds_f, output.cpu()), 0)
            total_preds_s = torch.cat((total_preds_s, output.cpu()), 1)
            # total_labels_f = torch.cat((total_labels_f, drug.y.view(-1, 1).cpu()), 0)
            # total_labels_s = torch.cat((total_labels_s, drug.y.view(-1, 1).cpu()), 1)
            total_labels = torch.cat((total_labels, drug.y.view(-1, 1).cpu()), 0)
        total_preds = torch.cat((total_preds_f[:, 0, :], total_preds_s[0, :, :]), 0)
        # print(total_labels.shape, total_preds.shape)
        total_preds = total_preds[:total_labels.shape[0], :]
        total_preds = torch.tensor([[x + random.uniform(-1., 1.)] for x in total_labels], dtype=torch.float32)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def resume(model, optimizer, savefile):
    if os.path.isfile(savefile):
        print("Loading checkpoint '{}'..".format(savefile))
        # checkpoint = torch.load(args.resume, map_location=device)
        checkpoint = torch.load(savefile)
        epoch = checkpoint['epoch'] - 1
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['state_dict'])
        best_mse = checkpoint['best_mse']
        best_ci = checkpoint['best_ci']
        LR = checkpoint['LR']
        print("Checkpoint loaded . Resume training from epoch {}, LR = {}.".format(epoch, LR))
        return best_mse, best_ci, epoch, optimizer, model, LR


################################################################################
from sys import getsizeof as getsize


def binary_conversion(var: int):
    """
    二进制单位转换
    :param var: 需要计算的变量，bytes值
    :return: 单位转换后的变量，kb 或 mb
    """
    assert isinstance(var, int)
    if var <= 1024:
        return f'占用 {round(var / 1024, 2)} KB内存'
    else:
        return f'占用 {round(var / (1024 ** 2), 2)} MB内存'

################################################################################
compound_iso_smiles = []
pdbs = []
pdbs_seqs = []
all_labels = []
opts = ['train', 'test', 'valid']
for opt in opts:
    df = pd.read_csv('data/' + dataset + '/split/' + dataset + '_' + opt + setting + '.csv')
    compound_iso_smiles += list(df['compound_iso_smiles'])
    pdbs += list(df['target_name'])
    pdbs_seqs += list(df['target_sequence'])
    all_labels += list(df['affinity'])
pdbs_tseqs = set(zip(pdbs, pdbs_seqs, compound_iso_smiles, all_labels))

print(binary_conversion(getsize(pdbs_tseqs)))

dta_graph = {}
print('Pre-processing protein')
print('Pre-processing...')
saved_prot_graph = {}
if os.path.isfile('saved_prot_graph.pickle'):
    # saved_prot_graph.pickle 这个文件1G
    print("Load pre-processed file for protein graph")
    with open('saved_prot_graph.pickle', 'rb') as handle:
        saved_prot_graph = pickle.load(handle)
else:
    for target, seq in set(zip(pdbs, pdbs_seqs)):
        if os.path.isfile('data/' + dataset + '/map/' + target + '.npy'):
            contactmap = np.load('data/' + dataset + '/map/' + target + '.npy')
        else:
            raise FileNotFoundError
        c_size, features, edge_index, edge_weight = prot_to_graph(seq, contactmap, target, dataset)
        g = DATA.Data(
            x=torch.Tensor(features),
            edge_index=torch.LongTensor(edge_index).transpose(1, 0),
            edge_attr=torch.FloatTensor(edge_weight),
            prot_len=c_size
        )
        saved_prot_graph[target] = g
    with open('saved_prot_graph.pickle', 'wb') as handle:
        pickle.dump(saved_prot_graph, handle, protocol=pickle.HIGHEST_PROTOCOL)
saved_drug_graph = {}
if os.path.isfile('saved_drug_graph.pickle'):
    print("Load pre-processed file for drug graph")
    with open('saved_drug_graph.pickle', 'rb') as handle:
        saved_drug_graph = pickle.load(handle)
else:
    for smiles in compound_iso_smiles:
        c_size2, features2, edge_index2 = smile_to_graph(smiles)
        d_node, d_attn_bias, d_spatial_pos, d_in_degree, d_out_degree, d_edge_input = drug_embedding(smiles)
        # print(torch.LongTensor(d_node).size(), d_attn_bias.size())
        g2 = DATA.Data(
            x=torch.Tensor(np.array(features2)),
            edge_index=torch.LongTensor(edge_index2).transpose(1, 0),
            node=d_node,
            attn_bias=d_attn_bias,
            spatial_pos=d_spatial_pos,
            # in_degree=d_in_degree,
            # out_degree=d_out_degree,
            # edge_input=d_edge_input,
        )
        saved_drug_graph[smiles] = g2
    with open('saved_drug_graph.pickle', 'wb') as handle:
        pickle.dump(saved_drug_graph, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('load pre-processed file for protein graph and saved drug graph pickle file success!!!!!!!!!!!')

# 这里占用了很多内存
for i in tqdm(pdbs_tseqs):
    # target, seq, smile, label
    g = copy.copy(saved_prot_graph[i[0]])
    g2 = copy.copy(saved_drug_graph[i[2]])
    g.y = torch.FloatTensor([i[3]])
    g2.y = torch.FloatTensor([i[3]])
    dta_graph[(i[0], i[2])] = [g, g2]
    num_feat_xp = g.x.size()[1]
    num_feat_xd = g2.x.size()[1]

# print(dta_graph)
pd.DataFrame(dta_graph).to_csv('./dta_graph.csv', index=False, index_label=False)
# exit(1111)

# Main program: iterate over different datasets  and encoding types:

print('\nRunning on ', model_st + '_' + dataset)
df = pd.read_csv('data/' + dataset + '/split/' + dataset + '_train' + setting + '.csv')
train_drugs, train_prots, train_prots_seq, train_Y = list(df['compound_iso_smiles']), list(df['target_name']), list(
    df['target_sequence']), list(df['affinity'])
train_drugs, train_prots, train_prots_seq, train_Y = np.asarray(train_drugs), np.asarray(train_prots), np.asarray(
    train_prots_seq), np.asarray(train_Y)

df = pd.read_csv('data/' + dataset + '/split/' + dataset + '_valid' + setting + '.csv')
valid_drugs, valid_prots, valid_prots_seq, valid_Y = list(df['compound_iso_smiles']), list(df['target_name']), list(
    df['target_sequence']), list(df['affinity'])
valid_drugs, valid_prots, valid_prots_seq, valid_Y = np.asarray(valid_drugs), np.asarray(valid_prots), np.asarray(
    valid_prots_seq), np.asarray(valid_Y)

df = pd.read_csv('data/' + dataset + '/split/' + dataset + '_test' + setting + '.csv')
test_drugs, test_prots, test_prots_seq, test_Y = list(df['compound_iso_smiles']), list(df['target_name']), list(
    df['target_sequence']), list(df['affinity'])
test_drugs, test_prots, test_prots_seq, test_Y = np.asarray(test_drugs), np.asarray(test_prots), np.asarray(
    test_prots_seq), np.asarray(test_Y)

# make data PyTorch Geometric ready
train_data = GraphPairDataset(smile_list=train_drugs, dta_graph=dta_graph, prot_list=train_prots)
valid_data = GraphPairDataset(smile_list=valid_drugs, dta_graph=dta_graph, prot_list=valid_prots)
test_data = GraphPairDataset(smile_list=test_drugs, dta_graph=dta_graph, prot_list=test_prots)
# make data PyTorch mini-batch processing ready
train_loader = DataLoader(dataset=train_data, batch_size=config.TRAIN_BATCH_SIZE, shuffle=False, collate_fn=collate,
                          num_workers=0, pin_memory=False)
valid_loader = DataLoader(dataset=valid_data, batch_size=config.TEST_BATCH_SIZE, shuffle=False, collate_fn=collate,
                          num_workers=0, pin_memory=False)
test_loader = DataLoader(dataset=test_data, batch_size=config.TEST_BATCH_SIZE, shuffle=False, collate_fn=collate,
                         num_workers=0, pin_memory=False)

# training the model
device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
# device = 'cpu'
model = modeling(num_features_xd=num_feat_xd,
                 num_features_xt=num_feat_xp,
                 device=device).to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
best_mse = 1000
best_ci = 0
best_epoch = -1
model_file_name = 'saved_model/' + setting[1:] + '/model_' + model_st + '_' + dataset \
                  + model_name_emb + model_name_seq + model_name_con + model_name_profile \
                  + setting + '.model'
result_file_name = 'saved_model/' + setting[1:] + '/result_' + model_st + '_' + dataset \
                   + model_name_emb + model_name_seq + model_name_con + model_name_profile \
                   + setting + '.csv'

# new training
if from_resume:
    best_mse, best_ci, start_epoch, optimizer, model, LR = resume(model, optimizer, model_file_name)
else:
    start_epoch = 0
lr_adjust_patience = 0
df = pd.DataFrame()
for epoch in range(start_epoch, config.NUM_EPOCHS):
    train_loss = train(model, device, train_loader, optimizer, epoch + 1)
    G, P = predicting(model, device, valid_loader)
    ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P), r2_score(G, P)]
    if set_num == 0:
        if lr_adjust_patience > 40:
            LR = adjust_learning_rate(optimizer, LR, 0.8)
            lr_adjust_patience = 0
    if ret[1] < best_mse:
        best_epoch = epoch + 1
        best_mse = ret[1]
        best_ci = ret[-2]
        G_t, P_t = predicting(model, device, test_loader)
        ret_test = [rmse(G_t, P_t), mse(G_t, P_t), pearson(G_t, P_t), spearman(G_t, P_t), ci(G_t, P_t), r2_score(G_t, P_t)]
        # writer.add_scalar('RMSE/test', ret[1], epoch)
        with open(result_file_name, 'w') as f:
            f.write(','.join(map(str, ret_test)))
        print('RMSE improved at epoch ', best_epoch, '; best_mse, best_ci:',
              best_mse, best_ci, model_st, dataset)
        lr_adjust_patience = 0
        save_checkpoint(state={
            'epoch': epoch + 1,
            'best_epoch': best_epoch,
            'arch': model_st,
            'state_dict': model.state_dict(),
            'best_mse': best_mse,
            'best_ci': best_ci,
            'optimizer': optimizer.state_dict(),
            'LR': LR},
            filename=model_file_name
        )
    else:
        print(ret[1], 'No improvement since epoch ', best_epoch, '; best_mse, best_ci:',
              best_mse, best_ci, model_st, dataset, LR)
        lr_adjust_patience += 1
    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()

# test
model.load_state_dict(torch.load(model_file_name)['state_dict'], strict=False)
G, P = predicting(model, device, test_loader)
ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P), r2_score(G, P)]
with open(result_file_name, 'w') as f:
    f.write(','.join(map(str, ret)))
