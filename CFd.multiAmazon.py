import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import time
import argparse
import random
import torch
import pickle
import numpy as np
import os
from  torch.nn.utils.rnn import pad_sequence  
import sys
from torch.autograd import Variable



seed = 3473497
torch.cuda.manual_seed(seed)
np.random.seed(seed * 13 // 7)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
	print ('GPU will be used')
else:
	print ("CPU will be used")

def get_parser():
	# parse parameters
	parser = argparse.ArgumentParser()
	parser.add_argument("--save_path", type=str,
                        help="Experiment dump path")
	parser.add_argument("--embed_size", type=int, default = 1024)
	parser.add_argument("--hidden_size", type=int, default = 512)
	parser.add_argument("--neg_n", type=int, default = 5)
	parser.add_argument("--nclass", type = int, default = 2)
	parser.add_argument("--ndomain", type = int, default = 2)
	parser.add_argument("--batch_size", type = int, default = 16)
	parser.add_argument("--text_batch_size", type = int, default = 16)
	parser.add_argument("--train_path", type = str)
	parser.add_argument("--test_path", type = str)
	parser.add_argument("--valid_path", type = str)
	parser.add_argument("--raw_train_path", nargs = '+',type = str)	
	parser.add_argument("--raw_valid_path", type = str)
	parser.add_argument("--save_mi_path", type = str)
	parser.add_argument("--mode", type = str, )
	parser.add_argument("--n_epochs", type = int, default = 30)
	parser.add_argument("--lr", type = float, default = 0.1)
	parser.add_argument("--text_lr", type = float, default = 0.1)
	parser.add_argument("--random_lstm", type=int, default = 0)
	parser.add_argument("--alpha", type = float, default = 1.)
	parser.add_argument("--mi_lamda_s", type = float, default = 0)
	parser.add_argument("--mi_lamda_t", type = float, default = 1.)
	parser.add_argument("--mi_lamda_t_class", type = float, default = 1.)
	parser.add_argument("--lamda", type = float, default = 1.)
	parser.add_argument("--entropy_lamda", type = float, default = 1.)
	parser.add_argument("--contract_lamda", type = float, default = 1.)
	parser.add_argument("--yu", type = float, default = 0.3)
	parser.add_argument("--topk", type = int, default = 10)
	parser.add_argument("--pseudo_t_lamda", type = float, default = 1.)
	parser.add_argument("--inter_time", type = int, default = 5)
	parser.add_argument("--cluster_lamda", type = float, default = 1.)
	parser.add_argument("--num_to_return", type = int, default = 3000)
	parser.add_argument("--intra_loss", type = float, default = 1.)
	parser.add_argument("--domain_mode", type = str)


	return parser



#*--- Data Loader ---*#
def load_train(filename):
	## label \t text
	data = []
	with open(filename, 'r') as f:
		for line in f:
			data.append(line.strip().split('\t')) 
		print ("number of training pairs is", len(data))
	return data

def load_valid(filename):
	data = []
	with open(filename, 'r') as f:
		for line in f:
			data.append(line.strip().split('\t'))
		print ("number of valid pairs is", len(data))
	return data

def load_test(filename):
	data = []
	with open(filename, 'r') as f:
		for line in f:
			data.append(line.strip().split('\t'))
		print ("number of test pairs is", len(data))	
	return data

def load_raw(filename):
	data = []
	with open(filename, 'r') as f:
		for line in f:
			data.append([0, line.strip()])
		print ("number of raw pairs is", len(data))

	return data

#*--- Model ---*#

def xlm_r(data):
	"""
	generate xlm-r features
	"""
	xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.large').to(device)
	xlmr.eval()
	embeded_data = []
	with torch.no_grad():
		for pair in data:
			#print (pair)
			if len(pair) == 1:
				continue
			tokens = xlmr.encode(pair[1])
			if list(tokens.size())[0] > 512:
				tokens = tokens[:512]
			#last_layer_features = xlmr.extract_features(tokens) # 1 * length * embedding_size
			#mean_features = torch.mean(last_layer_features, dim = 1).view(-1) # 1 * embedding_size
			topk_layers_features = xlmr.extract_features(tokens, return_all_hiddens=True) # 1 * length * embedding_size
			topk_layers_features = topk_layers_features[(len(topk_layers_features) - params.topk):]
			mean_features = [torch.mean(layer_feature, dim = 1).view(-1) for layer_feature in topk_layers_features]

			label = float(pair[0])   #torch.tensor([[float(pair[0])]]).to(device)
			new_pair = [label, mean_features]
			embeded_data.append(new_pair)
		return embeded_data


class FC(nn.Module):
	def __init__(self, hidden_size, num_class):
		super().__init__()
		self.fc = nn.Linear(hidden_size, num_class)
		self.init_weights()		

	def init_weights(self):
		initrange = 0.01
		self.fc.weight.data.uniform_(-initrange, initrange)
		self.fc.bias.data.zero_()

	def forward(self, feature):
		return self.fc(torch.tanh(feature))

class MMD_loss(nn.Module):
	def __init__(self, kernel_mul = 2.0, kernel_num = 5):
		super(MMD_loss, self).__init__()
		self.kernel_num = kernel_num
		self.kernel_mul = kernel_mul
		self.fix_sigma = None
		#return
	def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
		n_samples = int(source.size()[0])+int(target.size()[0])
		total = torch.cat([source, target], dim=0)

		total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
		total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
		L2_distance = ((total0-total1)**2).sum(2)
		if fix_sigma:
			bandwidth = fix_sigma
		else:
			bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
		bandwidth /= kernel_mul ** (kernel_num // 2)
		bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
		kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
		return sum(kernel_val)

	def forward(self, source, target):
		batch_size = int(source.size()[0])
		kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
		XX = kernels[:batch_size, :batch_size]
		YY = kernels[batch_size:, batch_size:]
		XY = kernels[:batch_size, batch_size:]
		YX = kernels[batch_size:, :batch_size]
		loss = torch.mean(XX + YY - XY -YX)
		return loss



def sharpen(output, t):
	# output: batch, nclass
	output = torch.pow(output, t)
	return output / torch.sum(output, dim = 1).view(-1, 1, 1)

	
class Ave(nn.Module):
	def __init__(self, embed_size, hidden_size):
		super().__init__()
		self.fc0 = nn.Linear(embed_size, 512)
		self.fc = nn.Linear(embed_size, hidden_size)
		self.att = nn.Linear(hidden_size, 1)#(embed_size, 1)
		self.init_weights()
	def init_weights(self):
		initrange = 0.01
		self.fc0.weight.data.uniform_(-initrange, initrange)
		self.fc0.bias.data.zero_()
		self.fc.weight.data.uniform_(-initrange, initrange)
		self.fc.bias.data.zero_()
		self.att.weight.data.uniform_(-initrange, initrange)
		self.att.bias.data.zero_()
	def forward(self, text):#, return_att = False):
		batch, N, dim = text.size()
		
		if N == 1:
			return self.fc(torch.tanh(text.view(batch, dim)))

		text = self.fc(torch.tanh(text))
		batch, N, dim = text.size()

		att_weight = torch.tanh(self.att(text)).view(batch, N, 1) # batch, N, 1
		att_weight = F.softmax(att_weight, dim = 1)
		att_weight = sharpen(att_weight, 1/params.yu)

		weighted_text = torch.sum( text * att_weight,  dim = 1 ).view(batch, dim)

		return weighted_text




# L2 loss
class Anchor1(nn.Module):
	def __init__(self, hidden_size, nclass):
		super().__init__()
		self.cos = nn.CosineSimilarity(dim = 1, eps = 1e-6)

	def init_weights(self):
		initrange = 0.01
	def forward(self, feat, centers, index):
		batch, dim = feat.size()
		new_centers = [] 
		for i in range(batch):
			c_id = index[i].item()
			new_centers.append(centers[:, c_id].view(1, -1))
		new_centers = torch.cat(new_centers, dim = 0)
		inter_loss = torch.mean( torch.sum(  (feat - new_centers) ** 2, dim = 1) )
		return params.intra_loss *  inter_loss   #cos_output




class Large(nn.Module):
	def __init__(self, input_size, hidden_size):
		super().__init__()
		self.fc = nn.Linear(hidden_size, input_size)
		self.init_weights()
	def init_weights(self):
		initrange = 0.01
		self.fc.weight.data.uniform_(-initrange, initrange)
		self.fc.bias.data.zero_()
	def forward(self, feature):
		return self.fc(torch.tanh(feature))


def generate_batch(batch):
	#if params.nclass == 5:		
	#	label = torch.LongTensor([entry[0]-1 for entry in batch])
	#else:
	label = torch.LongTensor([entry[0] for entry in batch])

	N_dim_ = []
	start = 10 - params.topk

	for entry in batch:
		if params.topk != 1:
			N_dim_.append(torch.cat([en.view(1, -1) for en in  entry[1][start:]], dim = 0))
		elif params.topk == 1:
			N_dim_.append(entry[1][-1].view(1, -1))

	size = N_dim_[0].size()
	text = torch.cat([en.view(1, size[0], size[1]) for en in N_dim_], dim = 0)

	return text, label

def train_func(params, lstm, fc, criterion, optimizer, train_set):
	# train the model
	train_loss = 0
	train_acc = 0
	data = DataLoader(train_set, batch_size = params.text_batch_size, shuffle = True, 
				collate_fn = generate_batch)

	for i, (text, cls) in enumerate(data):
		optimizer.zero_grad()

		text, cls = text.to(device), cls.to(device)

		output = lstm(text)
		output = fc(output)


		loss = criterion(output, cls)
		train_loss += loss.item()
		loss.backward()
		optimizer.step()
		train_acc += (output.argmax(1) == cls).sum().item()

	return train_loss / len(train_set), train_acc / len(train_set)

def test(params, lstm, fc, criterion, data_):
	loss = 0
	acc = 0
	data = DataLoader(data_, batch_size = params.batch_size, collate_fn = generate_batch)
	for text, cls in data:
		text, cls = text.to(device), cls.to(device)
		with torch.no_grad():
			output =  lstm(text)	
			output = fc(output)
			loss = criterion(output, cls)
			loss += loss.item()
			acc += (output.argmax(1) == cls).sum().item()
	return loss / len(data_), acc / len(data_)




class GradientReverse(torch.autograd.Function):
	scale = 1.0
	@staticmethod
	def forward(ctx, x):
		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		return GradientReverse.scale * grad_output.neg()
    
def grad_reverse(x, scale=1.0):
	GradientReverse.scale = scale
	return GradientReverse.apply(x)





def get_data_per_class(train_set):
	set_per_class = {'0': [], '1': [], '2':[]}
	for d in train_set:
		if d[0] == 0:
			set_per_class['0'].append(d)
		elif d[0] == 1:
			set_per_class['1'].append(d)
		else:
			set_per_class['2'].append(d)
	return set_per_class



def get_centers(train_set, lstm):
	with torch.no_grad():
		set_per_class = get_data_per_class(train_set)
		centers = []
		
		for i in range(params.nclass):
			print (i)
			print (len(set_per_class[str(i)]))

			set_ = DataLoader(set_per_class[str(i)], batch_size = len(set_per_class[str(i)]), collate_fn = generate_batch)
			for se, _ in set_:
				se = se.to(device)
				cen = torch.mean(lstm(se), dim = 0).view(-1, 1)
				centers.append(cen)
		return torch.cat(centers, dim = 1)



def pred_label(data_, lstm, fc,  num_to_return):
	data = DataLoader(data_, batch_size = len(data_), collate_fn = generate_batch)
	for text, cls in data:
		text, cls = text.to(device), cls.to(device)
		print ('in program of pred_label()')
		print ('size of text is %d' % (len(text)))
		with torch.no_grad():
			output = F.softmax(fc(lstm(text)), dim = 1)
			pred_raw = output.argmax(1)

			cross_entropy_raw = torch.sum(output * output.log(), dim = 1)
			cross_entropy_raw_sorted, indices = torch.sort(cross_entropy_raw, dim = -1, descending = True)

			all_data = [[]] * params.nclass
			pos = []
			neg = []
			neu = []
			for i in range(len(data_)):
				a = indices[i].item()
				c_d = [pred_raw[a], text[a]]
				if  int(  pred_raw[a].item()) == 0:
					neg.append(c_d)
				elif int(pred_raw[a].item()) == 1:
					pos.append(c_d)
				else:
					neu.append(c_d)
			all_data = [neg, pos, neu]
			return_data = []

			num_record = [0] * params.nclass
			labels = []
			for i in range(num_to_return):
				for k in range(params.nclass):
					j = (i + k) % params.nclass
					if num_record[j] < len(all_data[j]):
						labels.append(all_data[j][int(num_record[j])][0].item())
						return_data.append( all_data[j][int(num_record[j])] )
						num_record[j] += 1
						break
			#print (labels)
			#print ('from all the data, pos is: %d, neg is: %d, neu is: %d' % (len(pos), len(neg), len(neu)))
			#print ('size of returned data: %d'% (len(return_data)))

			random.shuffle(return_data)
			return return_data[:]#[:300]




#*--- MI training ---*#
def train_func_MI(params, lstm,  large, fc, domain_fc, criterion, optimizer, raw_train_set, train_set, teach_lstm = None, teach_fc = None):
	train_loss = 0	
	domain_loss = 0
	output_cluster_loss = 0


	mmd = MMD_loss()


	if teach_lstm != None:
		raw_set = pred_label(raw_train_set[1], teach_lstm, teach_fc, params.num_to_return)
		centers = get_centers(train_set + raw_set, lstm )#+ raw_set, lstm)
		anchor = Anchor1(params.embed_size, params.nclass)

	data = []
	for train in raw_train_set:
		data.append(DataLoader(train, batch_size = params.batch_size, shuffle=True, collate_fn = generate_batch))
	

	enu_data = []
	for d in data[0]:
		enu_data.append(d)
	data[0] = enu_data
	enu_data = []
	for d in data[1]:
		enu_data.append(d)
	data[1] = enu_data
	

	source_labeled_data = DataLoader(train_set, batch_size = params.text_batch_size, shuffle = True, collate_fn = generate_batch)
	enu_data = []
	for d in source_labeled_data:
		enu_data.append(d)
	source_labeled_data = enu_data


	if teach_lstm != None:	
		raw_set_batch = int(float(len(raw_set)) * params.text_batch_size / len(train_set))
		raw_set = DataLoader(raw_set, batch_size = raw_set_batch, shuffle = True, collate_fn = generate_batch)
		enu_data = []
		for d in raw_set:
			enu_data.append(d)
		raw_set = enu_data
	else:
		raw_set = source_labeled_data	

	

	random.shuffle(source_labeled_data)
	random.shuffle(data[0])
	random.shuffle(data[1])
	random.shuffle(raw_set)

	print (len(data[0]))
	print (len(data[1]))
	print (len(source_labeled_data))
	print (len(raw_set))


 
	for (text1, cls1), (text2, cls2), (text3, cls3), (text4, cls4) in zip(data[0], data[1], source_labeled_data, raw_set):
		batch_ul, N_, dim_ = text1.size()
		batch_l, _, _ = text3.size()
		optimizer.zero_grad()
		text1, cls1 = text1.to(device), cls1.to(device)
		text2, cls2 = text2.to(device), cls2.to(device)
		text3, cls3 = text3.to(device), cls3.to(device)
		text4, cls4 = text4.to(device), cls4.to(device)


		z1 = lstm(text1) #(batch, hidden_size)
		z2 = lstm(text2)
		z3 = lstm(text3)
		z4 = lstm(text4)



		# 1. cluster, 2. pesudo loss
		if teach_lstm != None and params.cluster_lamda >= 0:
			output_z4 = anchor(z4, centers, cls4) #pred_t_ul)
			output_z3 = anchor(z3, centers, cls3)
			cluster_loss = output_z4 + output_z3 #+ anchor(z2, centers_t, pred_t_ul) + anchor(z3, centers_s, cls3)   #+= criterion(output_z3, cls3)	
			output_cluster_loss += cluster_loss

			if params.pseudo_t_lamda >= 0:
				output_z4_super = fc(z4)
				pseudo_t_loss = criterion(output_z4_super, cls4) #pred_t_ul)

	
		text1 = torch.sum(text1, dim = 1).view(batch_ul, dim_)
		text2 = torch.sum(text2, dim = 1).view(batch_ul, dim_)
		text3 = torch.sum(text3, dim = 1).view(batch_l, dim_)


	
		neg_n = params.neg_n

		# I (x1, z1) + I (x1, z2)
		cos = nn.CosineSimilarity(dim = 1, eps = 1e-6)
		# I (x1, z1)
		z_ave_1 = text1
		z_z_ave_1_score = cos(large[0](z1), z_ave_1).view(-1, 1)
		z_z_ave_1_shuffle_score = []


		z_ave_2 = text2
		z_z_ave_2_score = cos(large[0](z2), z_ave_2).view(-1,1)
		z_z_ave_2_shuffle_score = []

		

		local_loss = 0
		if params.mi_lamda_t != 0 and params.mi_lamda_s != 0:
			for i in range(neg_n):
				r = torch.randperm(z1.size(0))
				z_z_ave_1_shuffle_score.append(cos(large[0](z1), z_ave_1[r]).view(-1, 1))   #(global_dis(z, z_ave[r]).view(-1,1))
			
				r = torch.randperm(z2.size(0))	
				z_z_ave_2_shuffle_score.append(cos(large[0](z2), z_ave_2[r]).view(-1, 1))

				local_loss += -torch.mean(z_z_ave_1_score - z_z_ave_1_shuffle_score[i]) * params.mi_lamda_s - torch.mean(z_z_ave_2_score - z_z_ave_2_shuffle_score[i]) * params.mi_lamda_t


		elif params.mi_lamda_t == 0 and params.mi_lamda_s != 0:
			for i in range(neg_n):
				r = torch.randperm(z1.size(0))
				z_z_ave_1_shuffle_score.append(cos(large[0](z1), z_ave_1[r]).view(-1, 1))   #(global_dis(z, z_ave[r]).view(-1,1))
				local_loss += -torch.mean(z_z_ave_1_score - z_z_ave_1_shuffle_score[i]) * params.mi_lamda_s 
		elif params.mi_lamda_t != 0 and params.mi_lamda_s == 0:
			for i in range(neg_n):
				r = torch.randperm(z2.size(0))
				z_z_ave_2_shuffle_score.append(cos(large[0](z2), z_ave_2[r]).view(-1, 1))
				ll = (z_z_ave_2_score - z_z_ave_2_shuffle_score[i]) 
				local_loss += -torch.mean(ll) * params.mi_lamda_t
			

		
		if params.lamda != 0:
			if params.domain_mode == 'kl':
				z_s = torch.mean(z1, dim = 0).view(-1)
				z_s = F.softmax(z_s, -1)
				z_t = torch.mean(z2, dim = 0).view(-1)
				z_t = F.softmax(z_t, -1)
				div_loss = torch.nn.KLDivLoss(size_average=True)(z_s.log(), z_t)\
                                	 + torch.nn.KLDivLoss(size_average=True)(z_t.log(), z_s)
			elif params.domain_mode == 'mmd':
				z_s = z1
				z_t = z2
				div_loss = mmd(z_s, z_t)
			elif params.domain_mode == 'adv':
				z_s = grad_reverse(z1)
				z_t = grad_reverse(z2)
				domain_output_s = domain_fc(z_s)
				domain_output_t = domain_fc(z_t)
				div_loss = criterion(domain_output_s, cls1) + criterion(domain_output_t, cls2)
		else:
			div_loss = 0
		
	
		### supervised loss
		output = fc(z3)
		super_loss = criterion(output, cls3)
		

		if teach_lstm == None:
			whole_loss = local_loss + params.lamda * div_loss  + super_loss #+ entropy_loss * params.entropy_lamda #+ whole_contract_loss * params.contract_lamda
		else:
			whole_loss = local_loss + params.lamda * div_loss + super_loss + pseudo_t_loss * params.pseudo_t_lamda + cluster_loss * params.cluster_lamda

		if params.mi_lamda_t == 0 and params.mi_lamda_s == 0:
			train_loss = 0
		else:
			train_loss +=  local_loss.item()  #(local_loss.item() -  params.alpha * loss_d.item())
		if params.lamda == 0:
			domain_loss = 0
		else:
			domain_loss += div_loss.item()
		whole_loss.backward()
		optimizer.step()

	if teach_lstm != None:
		print ('cluster_loss: %lf' % output_cluster_loss.item())


	return train_loss / len(train_set), domain_loss / len(train_set)





def initialize(params):
	fc = FC(params.hidden_size, params.nclass)
	return fc.to(device)


###### initialize an instance and train the model
def train(params):
	fc = initialize(params)
	random_lstm = params.random_lstm
	if  random_lstm == 0:
		lstm = load(params.save_mi_path)
		lstm.train()
		optimizer = torch.optim.Adam(fc.parameters(), lr = params.text_lr)
		print ('\t pre-trained lstm is loaded')
	else:
		lstm = Ave(params.embed_size, params.hidden_size).to(device)
		optimizer = torch.optim.Adam(list(fc.parameters()) + list(lstm.parameters()), lr = params.text_lr)
		print ('\t lstm is randomly initialized')
		lstm.train()
	
	n_epochs = params.n_epochs
	min_valid_loss = float('inf')


	criterion = torch.nn.CrossEntropyLoss().to(device)



	best_valid_loss = 0
	best_test_acc = 0	
	best_valid_acc = 0.
	patience = 0
	num = 10


	output_training_loss = []
	for epoch in range(n_epochs):
		start_time = time.time()
		train_loss, train_acc = train_func(params, lstm, fc, criterion, optimizer, train_set)
		fc.eval()
		lstm.eval()

		output_training_loss.append(train_loss)

		valid_loss, valid_acc = test(params, lstm, fc, criterion, valid_set)
		test_loss, test_acc = test(params, lstm, fc, criterion, test_set)
		fc.train()		
		lstm.train()

		if epoch == 0:
			best_valid_loss = valid_loss
			best_test_acc = test_acc

		secs = int(time.time() - start_time)
		mins = secs / 60
		secs = secs % 60

		print ('Epoch: %d' % (epoch + 1), " | time in %d minutes, %d seconds" % (mins, secs))
		print (f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
		print (f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')


		if best_valid_loss > valid_loss:
			best_valid_loss = valid_loss
			print (f'\tNew best valid loss:{best_valid_loss:.4f}')
			best_test_acc = test_acc
			patience = 0
		else:
			patience += 1
		if patience >= num:
			lr = optimizer.param_groups[0]['lr'] * 0.5
			for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
			patience = 0
			print (f'\tlr decay...')
		print (f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)')
		print (f'\t\tCurrent best test acc:{best_test_acc:.4f}')
	print (f'\tBest test acc:{best_test_acc:.4f}')
	print (output_training_loss)
	return best_test_acc

#*--- MI training ---*#

def save(params, lstm, path):
	print('save model parameters to [%s]' % path, file=sys.stderr)
	params = {
	    'args': params ,
            'lstm_state_dict': lstm.state_dict(),
        }
	torch.save(params, path)

def load(path, if_load_global_dic = False):
	data = torch.load(path)
	lstm = Ave(params.embed_size, params.hidden_size)
	lstm.load_state_dict(data['lstm_state_dict'])

	return lstm.to(device)
	



###### initialize an instance and train the model
def train_MI(params, teach_lstm = None, teach_fc = None):
	n_epochs = params.n_epochs
	min_valid_loss = float('inf')

	fc = initialize(params)

	domain_fc = FC(params.hidden_size, 2).to(device)

	teach_lstm = Ave(params.embed_size, params.hidden_size).to(device)
	teach_fc = initialize(params)


	lstm = Ave(params.embed_size, params.hidden_size).to(device)

	large_1 = Large(params.embed_size, params.hidden_size).to(device)
	large_2 = Large(params.embed_size, params.nclass).to(device)
	large = [large_1, large_2]

	criterion = torch.nn.CrossEntropyLoss().to(device)

	optimizer = torch.optim.Adam([{'params': lstm.parameters()},  {'params':large_1.parameters()}, {'params': large_2.parameters()}, {'params': fc.parameters()}, {'params':domain_fc.parameters()}], lr = params.lr)




	best_valid_loss = 100000
	best_test_acc = 0	
	patience = 0
	num = 10


	output_train_loss = []
	output_valid_loss = []
	output_domain_loss = []



	text = {}
	text['best_valid_loss'] = 10000
	text['best_test_acc'] = 0
	text['best_valid_acc'] = 0.
	text['patience'] = 0
	text['num'] = 10
	text['output_training_loss'] = []


	N_epochs = 60


	text_best_acc = []
	text_valid_best_acc = []



	if_saved = False
	best_test_acc = 0
	output_valid_acc = []
	output_test_acc = []


	fc_best = initialize(params)
	lstm_best = Ave(params.embed_size, params.hidden_size).to(device)



	params.num_to_return = 1000
	do_pesudo = params.pseudo_t_lamda
	for epoch in range(n_epochs):
		start_time = time.time()
		
		params.num_to_return += 300 #200 #50
		if do_pesudo:
			params.pseudo_t_lamda = (float(epoch) / n_epochs)#**2  #2
			

	
		if lstm_best == None:	
			train_loss, domain_loss  = train_func_MI(params, lstm, large, fc, criterion,  optimizer, raw_train_set, train_set)
		else:
			train_loss, domain_loss  = train_func_MI(params, lstm, large, fc, domain_fc, criterion, optimizer, raw_train_set, train_set, lstm_best, fc_best)


		output_train_loss.append(train_loss)
		output_domain_loss.append(domain_loss)

		fc.eval()
		lstm.eval()


		valid_loss, valid_acc = test(params, lstm, fc, criterion, valid_set)
		test_loss, test_acc = test(params, lstm, fc, criterion, test_set)

		output_valid_acc.append(valid_acc)
		output_test_acc.append(test_acc)

		fc.train()
		lstm.train()


		output_valid_loss.append(valid_loss)
		
		secs = int(time.time() - start_time)
		mins = secs / 60
		secs = secs % 60

		print ('Epoch: %d' % (epoch + 1), " | time in %d minutes, %d seconds" % (mins, secs))
		print (f'\tLamda: {lamda} (div_loss\t)')
		print (f'\tDomain Loss: {domain_loss:.4f}(train\t)')
		print (f'\tLoss: {train_loss:.4f}(train)\t')
		print (f'\t\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')
		print (f'\t\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)')

		
		if best_valid_loss > valid_loss:
			best_valid_loss = valid_loss.item()
			print (f'\tNew best valid loss:{best_valid_loss:.4f}')
			print (f'\tSave current best model')
			fc_best.load_state_dict(fc.state_dict())
			lstm_best.load_state_dict(lstm.state_dict())
			if_saved = True
			patience = 0
			best_test_acc = test_acc
		else:
			patience += 1
		
		
		if patience >= num:
			lr = optimizer.param_groups[0]['lr'] * 0.5
			for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
			patience = 0
			print (f'\tlr decay...')
		print (f'\t\tCurrent best test acc:{best_test_acc * 100:.1f}')

		
	
	print (f'train_loss')	
	print (output_train_loss)
	print (f'valid_loss')
	print (output_valid_loss)
	print (f'domain loss')
	print (output_domain_loss)

	print (f'\t\tBest test acc:{best_test_acc * 100:.1f}')

	print ('valid_acc')
	print (output_valid_acc)
	print ('test_acc')
	print (output_test_acc)

	return [lstm_best, fc_best],  best_test_acc, best_valid_loss








if __name__ == '__main__':
	parser = get_parser()
	params = parser.parse_args()
	lamda = params.lamda


	raw_train_set = []
	for path in params.raw_train_path:
		if os.path.exists(path + '.xlmr.topk.base'):
			print ('load pre-generated raw-train-text features from ' + path + ' ...')
			c_raw_train_set = pickle.load(open(path + '.xlmr.topk.base', 'rb'))
			raw_train_set.append(c_raw_train_set)
			print ('pre-generated raw-train-text features loaded')
		else:
			print ('start to generate raw-train-text xlmr features...')
			c_raw_train_set = xlm_r(load_raw(path))
			#os.makedirs(path + '.xlmr')
			pickle.dump(c_raw_train_set, open(path + '.xlmr.topk.base', 'wb'))
			raw_train_set.append(c_raw_train_set)
			print ('raw-train-text xlmr features generation is done and saved.')

	
	### for amazon only
	raw_train_set_s = []

	for i in range(2):   #params.ndomain):
		c_raw_train_set = raw_train_set[i]
		for j in range(len(c_raw_train_set)):
			c_raw_train_set[j][0] = 0.
		raw_train_set_s += c_raw_train_set
	raw_train_set_t = []
	for i in range(2, 4):
		c_raw_train_set = raw_train_set[i]
		for j in range(len(c_raw_train_set)):
			c_raw_train_set[j][0] = 1.
		raw_train_set_t += c_raw_train_set

	random.shuffle(raw_train_set_s)
	random.shuffle(raw_train_set_t)
	raw_train_set = [raw_train_set_s, raw_train_set_t]
	### for amazon only



	if os.path.isfile(params.train_path + '.xlmr.topk.base'):
		train_set = pickle.load(open(params.train_path + '.xlmr.topk.base', 'rb'))
	else:
		train_set = xlm_r(load_train(params.train_path))
		pickle.dump(train_set, open(params.train_path+'.xlmr.topk.base', 'wb'))

	if os.path.isfile(params.valid_path + '.xlmr.topk.base'):
		valid_set = pickle.load(open(params.valid_path + '.xlmr.topk.base', 'rb'))
	else:
		valid_set = xlm_r(load_valid(params.valid_path))
		pickle.dump(valid_set, open(params.valid_path+'.xlmr.topk.base', 'wb'))

	if os.path.isfile(params.test_path + '.xlmr.topk.base'):
		test_set = pickle.load(open(params.test_path + '.xlmr.topk.base', 'rb'))
	else:
		test_set = xlm_r(load_test(params.test_path))
		pickle.dump(test_set, open(params.test_path+'.xlmr.topk.base', 'wb'))


	

	def set_hyper(c, p, mi, dm = None):
		params.cluster_lamda = c
		params.pseudo_t_lamda = p#1.
		params.mi_lamda_t = mi
		params.lamda = 0     # kl: 500, mmd: 1, adv: 0.01
		params.domain_mode = dm
	
	
		if dm == 'kl':
			params.lamda = 500
		elif dm == 'mmd':
			params.lamda = 1
		elif dm == 'adv':
			params.lamda = 0.01
		elif dm == None:
			params.lamda = 0
		else:
			assert 1 == 0





	def train_MI_mode():   #params.mode == 'train_MI':
		best_, loss_ = [], []
		for i in range(9):#params.inter_time):
			params.intra_loss = 2. #2.		

			#p
			if i == 0:
				set_hyper(c=0, p=1, mi=0, dm = None)
			# kl:
			if i == 1:	
				set_hyper(c=0, p=0, mi=0, dm = 'kl')
			# mmd:
			if i == 2:
				set_hyper(c=0, p=0, mi=0, dm = 'mmd')
			# adv:
			if i == 3:
				set_hyper(c=0, p=0, mi=0, dm = 'adv')
			# mi
			if i == 4:
				set_hyper(c=0, p=0, mi=1, dm = None)
			# mi + c:
			if i == 5:
				set_hyper(c=1, p=0, mi=1, dm = None)
			# mi + p:
			if i == 6:
				set_hyper(c=0, p=1, mi = 1, dm = None)
			# c + p:
			if i == 7:
				set_hyper(c=1, p = 1, mi = 0, dm = None)
			# mi + c + p
			if i == 8:
				set_hyper(c=1, p = 1, mi = 1, dm = None)
				

			[lstm, fc], best_test_acc, best_valid_loss = train_MI(params)#, teach_lstm, teach_fc)
			best_.append(best_test_acc)
			loss_.append(best_valid_loss)


		for i in range(len(best_)):			
			print ('epoch %d: acc, %lf; loss, %lf' % (i, best_[i], loss_[i]))
		
		return best_, loss_#, dd
	if params.mode == 'train_MI':
		acc_, valid_loss_ = [], []
		for i in range(5):
			best_, loss_ = train_MI_mode()
			acc_.append(best_)
			valid_loss_.append(loss_)
			#best_acc_.append(bb)
		for i in range(5):
			print ('inter: %d' % (i+1))
			print (acc_[i])
			print (valid_loss_[i])


		p_, kl_, mmd_, adv_, mi_, mi_c_, mi_p_, c_p_, mi_p_c_ = [], [], [], [], [], [], [], [], []

	
		#ini_, cluster, pseudo, cluster_pseudo = [], [], [], []
		for i in range(5):
			p_.append(acc_[i][0])
			kl_.append(acc_[i][1])
			mmd_.append(acc_[i][2])
			adv_.append(acc_[i][3])
			mi_.append(acc_[i][4])
			mi_c_.append(acc_[i][5])
			mi_p_.append(acc_[i][6])
			c_p_.append(acc_[i][7])
			mi_p_c_.append(acc_[i][8])

		#cluster, pseudo, cluster_pseudo = [], [], []
		
		print (params.train_path)
		print (params.test_path)
		
		print ('p')
		print (p_)
		print ('kl')
		print (kl_)
		print ('mmd')
		print (mmd_)
		print ('adv')
		print (adv_)
		print ('mi')
		print (mi_)
		print ('mi_c')
		print (mi_c_)
		print ('mi_p')
		print (mi_p_)
		print ('c_p')
		print (c_p_)
		print ('mi_p_c')
		print (mi_p_c_)

		print ('p: %lf' % (np.mean(p_)))
		print ('kl: %lf' % (np.mean(kl_)))
		print ('mmd: %lf' % (np.mean(mmd_)))
		print ('adv: %lf' % (np.mean(adv_)))
		print ('mi: %lf' % (np.mean(mi_)))
		print ('mi + c: %lf' % (np.mean(mi_c_)))
		print ('mi + p: %lf' % (np.mean(mi_p_)))
		print ('c + p: %lf' % (np.mean(c_p_)))
		print ('mi + c + p: %lf' % (np.mean(mi_p_c_)))
	

	
	if params.mode == 'train':
		best_test_acc = []
		for i in range(5):
			best_test_acc.append(train(params))
		print ('\n\n')
		print (params.train_path)
		print (params.test_path)
		print (best_test_acc)
		print ('test_acc: %.4f' %  np.mean(best_test_acc))
