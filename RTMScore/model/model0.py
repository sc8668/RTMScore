import torch as th
import torch.nn.functional as F
import dgl
import numpy as np
import random
import dgl.function as fn
from torch import nn
import pandas as pd


class ResBlock(nn.Module):
	def __init__(self, in_channels, dropout_rate=0.15):
		super(ResBlock, self).__init__()
		
		self.projectDown_node = nn.Linear(in_channels, in_channels//4)
		self.projectDown_edge = nn.Linear(in_channels, in_channels//4)
		self.bn1_node = nn.BatchNorm1d(in_channels//4)
		self.bn1_edge = nn.BatchNorm1d(in_channels//4)
		self.conv = ENConv(in_channels//4)
				
		self.projectUp_node = nn.Linear(in_channels//4, in_channels)
		self.projectUp_edge = nn.Linear(in_channels//4, in_channels)
		self.dropout = nn.Dropout(p=dropout_rate)
		self.bn2_node = nn.BatchNorm1d(in_channels)
		nn.init.zeros_(self.bn2_node.weight)
		self.bn2_edge = nn.BatchNorm1d(in_channels)
		nn.init.zeros_(self.bn2_edge.weight)
		
	def forward(self, bg):
		node_feats = bg.ndata['hh']
		edge_feats = bg.edata['ee']
		h_node = F.elu(self.bn1_node(self.projectDown_node(node_feats)))
		h_edge = F.elu(self.bn1_edge(self.projectDown_edge(edge_feats)))
		h_node, h_edge = self.conv(bg, h_node, h_edge)
		h_node = self.dropout(self.bn2_node(self.projectUp_node(h_node)))
		node_feats = F.elu(h_node + node_feats)
		
		h_edge = self.dropout(self.bn2_edge(self.projectUp_edge(h_edge))) 
		edge_feats = F.elu(h_edge + edge_feats)
		bg.ndata['hh'] = node_feats
		bg.edata['ee'] = edge_feats
		return bg


class ENConv(nn.Module):
	def __init__(self, in_channels):
		super(ENConv, self).__init__()
		self.edge_mlp_1 = nn.Sequential(
								nn.Linear(in_channels*3, in_channels), 
								nn.BatchNorm1d(in_channels), 
								nn.ELU()
								)
		self.edge_mlp_2 = nn.Sequential(
								nn.Linear(in_channels*2, in_channels), 
								nn.BatchNorm1d(in_channels), 
								nn.ELU()
								)
		self.node_mlp = nn.Sequential(
								nn.Linear(in_channels*2, in_channels), 
								nn.BatchNorm1d(in_channels), 
								nn.ELU()
								)
	
	def forward(self, bg, node_feats, edge_feats):
		with bg.local_scope():
			bg.ndata['h'] = node_feats
			bg.edata['e'] = edge_feats
			bg.apply_edges(lambda edges: {'ex': self.edge_mlp_1(th.cat([edges.src['h'], edges.dst['h'], edges.data['e']], dim=1))})
			bg.update_all(lambda edges: {'m': self.edge_mlp_2(th.cat([edges.src['h'], edges.data['ex']], dim=1))},
						fn.mean('m', 'h_mean'),
						apply_node_func= lambda nodes: {'ho': self.node_mlp(th.cat([nodes.data['h'], nodes.data['h_mean']], dim=1))})
			out_node_feats = bg.ndata['ho']
			out_edge_feats = bg.edata['ex']
			return out_node_feats, out_edge_feats
			

class TargetNet(nn.Module):
	def __init__(self, in_channels, edge_features=3, hidden_dim=128, residual_layers=20, dropout_rate=0.15):
		super(TargetNet, self).__init__()
        
		self.node_encoder = nn.Linear(in_channels, hidden_dim)
		self.edge_encoder = nn.Linear(edge_features, hidden_dim)        
		self.conv1 = ENConv(hidden_dim)
		self.conv2 = ENConv(hidden_dim)
		self.conv3 = ENConv(hidden_dim)
		layers = [ResBlock(in_channels=hidden_dim, dropout_rate=dropout_rate) for i in range(residual_layers)] 
		self.resnet = nn.Sequential(*layers)
	
	def forward(self, bg, node_feats, edge_feats):
		node_feats = self.node_encoder(node_feats)
		edge_feats = self.edge_encoder(edge_feats)
		node_feats, edge_feats = self.conv1(bg, node_feats, edge_feats)
		node_feats, edge_feats = self.conv2(bg, node_feats, edge_feats)
		node_feats, edge_feats = self.conv3(bg, node_feats, edge_feats)
		with bg.local_scope():
			bg.ndata['hh'] = node_feats
			bg.edata['ee'] = edge_feats
			bg = self.resnet(bg)
			return bg.ndata['hh'], bg.edata['ee']


class LigandNet(nn.Module):
	def __init__(self, in_channels, edge_features=10, hidden_dim=128, residual_layers=20, dropout_rate=0.15):
		super(LigandNet, self).__init__()
		
		self.node_encoder = nn.Linear(in_channels, hidden_dim)
		self.edge_encoder = nn.Linear(edge_features, hidden_dim)
		self.conv1 = ENConv(hidden_dim)
		self.conv2 = ENConv(hidden_dim)
		self.conv3 = ENConv(hidden_dim)
		layers = [ResBlock(in_channels=hidden_dim, dropout_rate=dropout_rate) for i in range(residual_layers)] 
		self.resnet = nn.Sequential(*layers)
	
	def forward(self, bg, node_feats, edge_feats):
		node_feats = self.node_encoder(node_feats)
		edge_feats = self.edge_encoder(edge_feats)
		node_feats, edge_feats = self.conv1(bg, node_feats, edge_feats)
		node_feats, edge_feats = self.conv2(bg, node_feats, edge_feats)
		node_feats, edge_feats = self.conv3(bg, node_feats, edge_feats)
		with bg.local_scope():
			bg.ndata['hh'] = node_feats
			bg.edata['ee'] = edge_feats
			bg = self.resnet(bg)
			return bg.ndata['hh'], bg.edata['ee']

			
class RTMScoreM(nn.Module):
	def __init__(self, lig_model, prot_model, in_channels, hidden_dim, n_gaussians, dropout_rate=0.15, disttype="min",
					dist_threhold=1000):
		super(RTMScoreM, self).__init__()
		
		self.lig_model = lig_model
		self.prot_model = prot_model
		self.MLP = nn.Sequential(nn.Linear(in_channels*2, hidden_dim), 
								nn.BatchNorm1d(hidden_dim), 
								nn.ELU(), 
								nn.Dropout(p=dropout_rate)
								) 
		self.disttype = disttype
		self.z_pi = nn.Linear(hidden_dim, n_gaussians)
		self.z_sigma = nn.Linear(hidden_dim, n_gaussians)
		self.z_mu = nn.Linear(hidden_dim, n_gaussians)
		self.atom_types = nn.Linear(in_channels, 17)
		self.bond_types = nn.Linear(in_channels*2, 4)
		
		self.dist_threhold = dist_threhold	
    
	def forward(self, bgp, bgl, bgpl):	
		h_l, _ = self.lig_model(bgl, bgl.ndata['feats'].float(), bgl.edata['feats'].float())
		h_p, _ = self.prot_model(bgp, bgp.ndata['feats'].float(), bgp.edata['feats'].float())
		with bgpl.local_scope():
			bgpl.srcdata['h'] = h_p
			bgpl.dstdata['h'] = h_l
			bgpl.apply_edges(lambda edges: {'e': self.MLP(th.cat([edges.src['h'], edges.dst['h']], dim=1))})
			C = bgpl.edata['e']
			
		# Outputs
		pi = F.softmax(self.z_pi(C), -1)
		sigma = F.elu(self.z_sigma(C))+1.1
		mu = F.elu(self.z_mu(C))+1
		atom_types = self.atom_types(h_l)
		bond_types = self.bond_types(th.cat([h_l[bgl.edges()[0]],h_l[bgl.edges()[1]]], axis=1))
		if self.disttype == "min":
			dist = bgpl.edges["pl"].data["mindist"]
		elif self.disttype == "ca":
			dist = bgpl.edges["pl"].data["cadist"]
		elif self.disttype == "center":
			dist = bgpl.edges["pl"].data["cedist"]
		elif self.disttype == "all":
			dist = th.cat(bgpl.edges["pl"].data["mindist"], bgpl.edges["pl"].data["cadist"], bgpl.edges["pl"].data["cedist"],dim=1)		
		else:
			raise ValueError('the "disttype"should be one of ["all", "min", "ca" and "center"]')
		return pi, sigma, mu, dist, atom_types, bond_types
		






