import sys
import os
import pickle
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
	print ('GPU will be used')
else:
	print ("CPU will be used")


xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.large').to(device)
xlmr.eval()

def xlm_r(data):
	#xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.large').to(device)
	#xlmr.eval()
	embeded_data = [] 
	with torch.no_grad():
		for i, pair in enumerate(data):
                        #print (pair)
			if len(pair) == 1:
				continue
			tokens = xlmr.encode(pair[1])
			if list(tokens.size())[0] > 512: 
				tokens = tokens[:512]
			topk_layers_features = xlmr.extract_features(tokens, return_all_hiddens=True) # 1 * length * embedding_size
			topk_layers_features = topk_layers_features[(len(topk_layers_features) - 10):]
			mean_features = [torch.mean(layer_feature, dim = 1).view(-1) for layer_feature in topk_layers_features]

			#last_layer_features = xlmr.extract_features(tokens) # 1 * length * embedding_size
			#mean_features = torch.mean(last_layer_features, dim = 1).view(-1) # 1 * embedding_size
			label = float(pair[0])   #torch.tensor([[float(pair[0])]]).to(device)
                       	#new_pair = torch.cat((label, mean_features), dim = 1)
			new_pair = [label, mean_features]
			embeded_data.append(new_pair)
			if i % 1000 == 0:
				print (i / len(data))
		return embeded_data

def load_train(filename):
        ## label \t text
        data = []
        with open(filename, 'r') as f:
                for line in f:
                        data.append(line.strip().split('\t'))
                print ("number of training pairs is", len(data))
        return data


def load_raw(filename):
	data = []
	with open(filename, 'r') as f:
		for line in f:
			data.append([0, line.strip()])
		print ("number of raw pairs is", len(data))
	return data


num = len(open(sys.argv[1], 'r').readline().strip().split('\t'))

if not os.path.isfile(sys.argv[1] + '.xlmr.topk.base') or True:
	if num == 2:
		train_set = xlm_r(load_train(sys.argv[1]))
	elif num == 1:
		#texts = load_raw(sys.argv[1])
		#inter = int(len(texts) / 10)
		#train_set = []
		#for i in range(11):
		#	start = i * inter
		#	end = (i+1) * inter
			#train_set += xlm_r(texts[start:end])
		#	pickle.dump(xlm_r(texts[start:end]), open(sys.argv[1]+'.'+str(i)+'.xlmr.topk.base', 'wb'))
		train_set = xlm_r(load_raw(sys.argv[1]))
	pickle.dump(train_set, open(sys.argv[1]+'.xlmr.topk.base', 'wb'))
