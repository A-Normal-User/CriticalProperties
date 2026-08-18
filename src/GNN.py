"""This module contains the class definition for all graph neural networks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import scatter


class GraphAttentionPooling(nn.Module):
    def __init__(self, n_features, key_dim):
        super(GraphAttentionPooling, self).__init__()

        self.n_features = n_features

        self.query_weight = nn.Parameter(torch.Tensor(n_features, key_dim))
        self.key_weight = nn.Parameter(torch.Tensor(n_features, key_dim))
        self.value_weight = nn.Parameter(torch.Tensor(n_features, n_features))

        nn.init.xavier_uniform_(self.query_weight)
        nn.init.xavier_uniform_(self.key_weight)
        nn.init.xavier_uniform_(self.value_weight)

    def get_attention_scores(self, node_out, batch):
        _ , n_features = node_out.size()
        device = node_out.device

        Q = torch.matmul(node_out, self.query_weight) 
        K = torch.matmul(node_out, self.key_weight)

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (n_features ** 0.5)

        # First, create a mask that identifies all pairs of nodes within the same graph
        # This will be used to filter out interactions between nodes of different graphs
        mask = (batch.unsqueeze(1) == batch.unsqueeze(0)).to(device)

        # Set attention scores for nodes in different graphs to -inf
        attention_scores[~mask] = float('-inf')

        # Apply softmax to the attention scores
        attention_scores = torch.softmax(attention_scores, dim=-1)

        return attention_scores

    def forward(self, node_out, batch):
        _, n_features = node_out.size()
        n_graphs = batch.max().item() + 1
        device = node_out.device

        V = torch.matmul(node_out, self.value_weight)

        attention_scores = self.get_attention_scores(node_out, batch)

        # Apply attention scores to the value matrix
        context_matrix = torch.matmul(attention_scores, V)

        # Sum the context matrix for each graph
        return scatter(context_matrix.view(-1, n_features), batch, dim=0, dim_size=n_graphs, reduce='sum')
class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, n_features, key_dim, num_pooling_heads):
        super(MultiHeadAttentionPooling, self).__init__()

        self.n_features = n_features
        self.num_heads = num_pooling_heads

        self.heads = nn.ModuleList([GraphAttentionPooling(n_features, key_dim) for _ in range(self.num_heads)])

    def forward(self, node_out, batch):
        # Apply each head to the node embeddings
        head_outputs = [head(node_out, batch) for head in self.heads]

        # Get the mean of the head outputs
        return torch.mean(torch.stack(head_outputs), dim=0)
class GNN_GAT(nn.Module):
    """Implementation of GAT"""

    def __init__(self, node_dim, edge_dim, conv_dim, heads=5, dropout=0.1, num_layers=3):
        """Initializes GAT model. Takes in node and edge dimensions conv_dim is the hidden dimension
        of the GAT convolutional layers. Heads is the number of attention heads to use in the GAT"""
        super().__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATv2Conv(node_dim, conv_dim, heads, edge_dim=edge_dim, dropout=dropout, concat=False, share_weights=True))

        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(conv_dim, conv_dim, heads, edge_dim=edge_dim, dropout=dropout, concat=False, share_weights=True))

    def forward(self, x, edge_index, edge_attr):

        device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        for conv in self.convs:
            conv.to(device)
            x = conv(x, edge_index, edge_attr)
            x = F.gelu(x)

        return x
class Head(nn.Module):
    """Prediction Head that is added to the end of the GNN. Takes in the pooled node embeddings
    and yields the Antoine parameters"""

    def __init__(self, input_dim, hidden_dim, num_hidden_layers, out_dim):
        super().__init__()

        # Input Layer
        layers = []
        #layers.append(nn.BatchNorm1d(input_dim))
        layers.append(nn.Linear(input_dim, hidden_dim, bias=True))
        layers.append(nn.GELU())

        # Hidden layers
        for _ in range(num_hidden_layers):
            #layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())

        # Output layer
        #layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.Linear(hidden_dim, out_dim))

        # Combine all layers
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(x)
class GNN(nn.Module):
    def __init__(self, node_dim, edge_dim,
                 conv_dim, hidden_dim, num_hidden_layers, 
                 dropout,
                 num_gnn_layers=3, gnn_heads = 5, pooling_heads = 1):
        super().__init__()

        # GNN that graphs are passed to
        self.gnn = GNN_GAT(node_dim, edge_dim, conv_dim, num_layers=num_gnn_layers, dropout=dropout, heads=gnn_heads)

        self.out_dim = conv_dim

        self.pooling_function = MultiHeadAttentionPooling(self.out_dim, key_dim=32, num_pooling_heads=pooling_heads) 

        # Head takes the final node embeddings and temperature and gives the Antoine parameters
        # Two dimensions added for number of H-Donors and H-Acceptors
        self.head = Head(self.out_dim +2, hidden_dim, num_hidden_layers, 1)

    def get_embedding(self, x, edge_index, edge_attr, numHDonors, numHAcceptors, batch):
        '''Returns the graph embedding for the given graph. This is the output of the GNN and pooling function.'''

        # Produces graph with generated embeddings
        gnn_out = self.gnn(x, edge_index, edge_attr)
        
        # Aggregates node embeddings to single node/vector.
        graph_out = self.pooling_function(gnn_out, batch)

        # Append number of H-Donors and H-Acceptors to the final graph embedding
        graph_out = torch.cat((graph_out, numHDonors.unsqueeze(1), numHAcceptors.unsqueeze(1)), dim=1)
        return graph_out

    def forward(self, x, edge_index, edge_attr, numHDonors, numHAcceptors, batch):
        
        # Produces graph with generated embeddings
        gnn_out = self.gnn(x, edge_index, edge_attr)
        
        # Aggregates node embeddings to single node/vector.
        graph_out = self.pooling_function(gnn_out, batch)

        # Append number of H-Donors and H-Acceptors to the final graph embedding
        graph_out = torch.cat((graph_out, numHDonors.unsqueeze(1), numHAcceptors.unsqueeze(1)), dim=1)
        
        return self.head(graph_out)

import torch
from rdkit import Chem
import numpy as np
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
possible_atom_list = ['B', 'Se', 'Sb', 'In', 'Cl', 'I', 'Ga', 'P', 'N', 'Hg', 'Ge', 'Al', 'C', 'Sn', 'V', 'Br', 'O', 'S', 'As', 'Pb', 'Si', 'F', 'Ti']
possible_hybridization = [Chem.rdchem.HybridizationType.S,
                          Chem.rdchem.HybridizationType.SP, 
                          Chem.rdchem.HybridizationType.SP2,
                          Chem.rdchem.HybridizationType.SP3]
possible_num_bonds = [0,1,2,3,4]
possible_num_Hs  = [0,1,2,3] 
possible_stereo  = [Chem.rdchem.BondStereo.STEREONONE,
                    Chem.rdchem.BondStereo.STEREOZ,
                    Chem.rdchem.BondStereo.STEREOE]

def one_of_k_encoding(x, allowable_set):
    """Apply onehot encoding to feature."""

    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def atom_feature(atom):
    """Extract atom features from instance of rdkit molecule. Check if atom has formal charge or radical electrons and throws error if this is the case."""
        
    symbol        = atom.GetSymbol()
    Type_atom     = one_of_k_encoding(symbol, possible_atom_list) # kept to detect atoms not in possible_atom_list and raise error
    
    # throw error if atom has formal charge
    # if atom.GetFormalCharge() != 0:
    #     raise Exception("Atom has formal charge!")
    
    # throw error if atom has radical electrons
    # if atom.GetNumRadicalElectrons() != 0:
    #     raise Exception("Atom has radical electrons!")
    
    Ring_atom     = [atom.IsInRing()]
    Aromaticity   = [atom.GetIsAromatic()]
    Hybridization = one_of_k_encoding(atom.GetHybridization(), possible_hybridization)
    Bonds_atom    = one_of_k_encoding(len(atom.GetNeighbors()), possible_num_bonds)
    num_Hs        = one_of_k_encoding(atom.GetTotalNumHs(), possible_num_Hs)
    

    results = Type_atom + Ring_atom + Aromaticity + Hybridization + Bonds_atom + num_Hs

    return np.array(results).astype(np.float32)

def bond_feature(bond):
    """Extract bond features from instance of rdkit molecule."""

    bt = bond.GetBondType()
    
    type_stereo = one_of_k_encoding(bond.GetStereo(), possible_stereo)
    
    # Bond level features
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()] + \
        type_stereo

    return np.array(bond_feats).astype(np.float32)
def mol_to_pyg(mol):
    """Convert rdkit mol object to pytorch Data object. If conversion fails, return None. If illegal atom properties are detected, throw error."""

    # Return None if rdkit fails to produce molecule
    if mol is None:
      return None
    
    # Return error if molecule does not contain at least one carbon atom
    # if not any([atom.GetSymbol() == 'C' for atom in mol.GetAtoms()]):
    #     raise Exception("Molecule does not contain at least one carbon atom.")

    numHDonors = AllChem.CalcNumHBD(mol)
    numHAcceptors = AllChem.CalcNumHBA(mol)

    # For a molecule get the bonded atoms
    id_pairs = ((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds())
    atom_pairs = [z for (i, j) in id_pairs for z in ((i, j), (j, i))]
    bonds = (mol.GetBondBetweenAtoms(i, j) for (i, j) in atom_pairs)
    
    # Construct node and edge features
    atom_features = np.array([atom_feature(a) for a in mol.GetAtoms()])
    bond_features = np.array([bond_feature(b) for b in bonds])

    # Create input data graph
    d = Data(edge_index=torch.tensor(np.array(list(zip(*atom_pairs))), dtype=torch.int64),
             x=torch.FloatTensor(atom_features), 
             edge_attr=torch.FloatTensor(bond_features),  #
             numHAcceptors=torch.tensor([numHAcceptors]),
             numHDonors=torch.tensor([numHDonors]))
    
    # Check if valid Data object. Raises error if not.
    d.validate()
    return d

def preprocess(smiles_list: list, batch_size: int = 32):
    """Preprocess the input data for the model."""
    mol_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    input_loader = DataLoader([mol_to_pyg(mol) for mol in mol_list], batch_size=batch_size)

    return input_loader

class GNN_Tc(torch.nn.Module):
    '''Module for the direct prediction of vapor pressures using GRAPPA.'''
    def __init__(self, GNN):
        super(GNN_Tc, self).__init__()
        self.model = GNN

    def forward(self, smiles_list: list):
        '''
        Gives a list of vapor pressures calculated with GRAPPA.
        Requires a list of smiles and the corresponding temperatures as input.
        '''
        input_loader = preprocess(smiles_list, 32)
        prediction_list = []
        for batch in input_loader:
            prediction_list.extend(self.model(batch.x, batch.edge_index, batch.edge_attr, batch.numHDonors, batch.numHAcceptors, batch.batch).detach().numpy())
        # Write prediction and unit into dictionary
        return np.exp(prediction_list)
    
    def get_embedding(self, smiles_list: list):
        '''
        Gives a list of graph embeddings from the GNN.
        Requires a list of smiles as input.
        '''
        input_loader = preprocess(smiles_list, 32)
        embedding_list = []
        for batch in input_loader:
            embedding_list.extend(self.model.get_embedding(batch.x, batch.edge_index, batch.edge_attr, batch.numHDonors, batch.numHAcceptors, batch.batch).detach().numpy())
        return np.array(embedding_list)