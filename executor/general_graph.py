import networkx as nx

import config
from graph_conversion import *
import algos
from preprocess import *


def sdf2graph(smile):
    drug = 'data/'+config.dataset+'/sdf/' + smile + '.sdf'
    mol = Chem.MolFromMolFile(drug)

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype=np.int64)
    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T
        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    # graph = dict()
    # graph['edge_index'] = edge_index
    # graph['edge_feat'] = edge_attr                  # dim = 3
    # graph['node_feat'] = x
    # graph['num_nodes'] = len(x)
    # return graph

    x = torch.from_numpy(x)
    edge_attr = torch.from_numpy(edge_attr)
    edge_index = torch.from_numpy(edge_index)
    return x, edge_attr, edge_index


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index


def prot_to_graph(seq, prot_contactmap, prot_target, dataset='davis'):
    c_size = len(seq)
    eds_seq = []
    if config.is_seq_in_graph:
        for i in range(c_size - 1):
            eds_seq.append([i, i + 1])
        eds_seq = np.array(eds_seq)
    eds_contact = []
    if config.is_con_in_graph:
        eds_contact = np.array(np.argwhere(prot_contactmap >= 0.5))

    # add an reserved extra node for drug node
    eds_d = []
    for i in range(c_size):
        eds_d.append([i, c_size])

    eds_d = np.array(eds_d)
    if config.is_seq_in_graph and config.is_con_in_graph:
        eds = np.concatenate((eds_seq, eds_contact, eds_d))
    elif config.is_con_in_graph:
        eds = np.concatenate((eds_contact, eds_d))
    else:
        eds = np.concatenate((eds_seq, eds_d))

    edges = [tuple(i) for i in eds]
    g = nx.Graph(edges).to_directed()
    features = []
    ss_feat = []
    sas_feat = []
    if config.is_profile_in_graph:
        ss_feat = aa_ss_feature(prot_target, dataset)
        sas_feat = aa_sas_feature(prot_target, dataset)
    sequence_output = np.load('data/'+dataset+'/emb/' + prot_target + '.npz', allow_pickle=True)
    sequence_output = sequence_output[prot_target].reshape(-1, 1)[0][0]['seq'][1:-1, :]
    sequence_output = sequence_output.reshape(sequence_output.shape[0], sequence_output.shape[1])
    for i in range(c_size):
        if config.is_profile_in_graph:
            if config.is_emb_in_graph:
                aa_feat = np.concatenate((np.asarray(sequence_output[i], dtype=float), ss_feat[i], sas_feat[i]))
            else:
                aa_feat = np.concatenate((aa_features(seq[i]), ss_feat[i], sas_feat[i]))
        else:
            if config.is_emb_in_graph:
                aa_feat = np.asarray(sequence_output[i], dtype=float)
            else:
                aa_feat = aa_features(seq[i])
        features.append(aa_feat)

    # place holder feature vector for drug
    place_holder = np.zeros(features[0].shape, dtype=float)
    features.append(place_holder)

    edge_index = []
    edge_weight = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        # if e1 == c_size or e2 == c_size:
        #     edge_weight.append(0.5)
        # else:
        edge_weight.append(1.0)
    return c_size, features, edge_index, edge_weight


def drug_embedding(smile, max_node=96):
    x, edge_attr, edge_index = sdf2graph(smile)
    # N = x.size(0)
    N = x if x.size(0) >= max_node else pad_2d_unsqueeze(x, max_node)
    # N = pad_2d_unsqueeze(x, max_node).squeeze(0)
    N = N.size(1)
    x = mol_to_single_emb(x)
    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = mol_to_single_emb(edge_attr) + 1

    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    spatial_pos = torch.from_numpy(shortest_path_result).long()
    attn_bias = torch.zeros(
        [N, N], dtype=torch.float)  # with graph token
    node = x
    attn_bias = attn_bias
    spatial_pos = spatial_pos
    in_degree = adj.long().sum(dim=1).view(-1)
    out_degree = adj.long().sum(dim=0).view(-1)
    edge_input = torch.from_numpy(edge_input).long()

    return node, attn_bias, spatial_pos, in_degree, out_degree, edge_input
