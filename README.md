# CFd
This is the implement of our paper: [Feature Adaptation of Pre-Trained Language Models across Languages and Domains with Robust Self-Training](https://www.aclweb.org/anthology/2020.emnlp-main.599.pdf)


## Environment
Python 3.7.4 

Pytorch 1.4.0

CUDA 8.0.61

## Data
The data can be downloaded from this link: [CFd-data](https://drive.google.com/file/d/1IDEzcCAPIQXGOc9cqeOUy5jG94bvIpu7/view?usp=sharing)

## How to run the code
for monoAmazon:

for example: book -> music

	1. generate last 10-layer features:
		python generate.xlmr.topk.py ./data/small/book/set1_text.txt 
		python generate.xlmr.topk.py ./data/small/music/set1_text.txt 
		python generate.xlmr.topk.py ./data/small/book/train.txt
		python generate.xlmr.topk.py ./data/small/book/test.txt
		python generate.xlmr.topk.py ./data/small/music/all.txt 

	2. run CFd.monoAmazon.py:
		python CFd.monoAmazon.py  --raw_train_path   ./data/small/book/set1_text.txt  ./data/small/music/set1_text.txt   --mode train_MI --n_epochs 35 --lr 0.0001 --text_lr 0.0001 --hidden_size 256  --batch_size 50 --text_batch_size 50 --neg_n 10 --nclass 3  --train_path  ./data/small/book/train.txt --valid_path  ./data/small/book/test.txt  --test_path  ./data/small/music/all.txt --save_mi_path path_to_save --random_lstm 1  --lamda 0  --mi_lamda_s 0 --mi_lamda_t 1  --pseudo_t_lamda 1  --cluster_lamda 1 --topk 10  --intra_loss 1

for multiAmazon:

for example:  en, books -> fr, books

	1. generate last 10-layer features:
		python generate.xlmr.topk.py ./data/sampled_raw_data/6000_en_books_raw.txt
		python generate.xlmr.topk.py ./data/annotated_data/en/books/train.txt
		python generate.xlmr.topk.py ./data/sampled_raw_data/6000_fr_books_raw.txt
		python generate.xlmr.topk.py ./data/annotated_data/fr/books/train.txt
		python generate.xlmr.topk.py ./data/annotated_data/fr/books/test.txt
		python generate.xlmr.topk.py ./data/annotated_data/en/books/test.txt

	2. run CFd.multiAmazon.py: 
		python CFd.multiAmazon.py --raw_train_path  ./data/sampled_raw_data/6000_en_books_raw.txt ./data/annotated_data/en/books/train.txt  ./data/sampled_raw_data/6000_fr_books_raw.txt ./data/annotated_data/fr/books/train.txt  --mode train_MI --n_epochs 20 --lr 0.0005 --text_lr 0.0005 --hidden_size 256 --batch_size 200 --text_batch_size 50 --neg_n 10 --nclass 2  --train_path ./data/annotated_data/en/books/train.txt --test_path ./data/annotated_data/fr/books/test.txt --valid_path ./data/annotated_data/en/books/test.txt --save_mi_path path_to_save --random_lstm 1 --lamda 5000 --mi_lamda_s 0 --mi_lamda_t 1 --pseudo_t_lamda 1  --cluster_lamda 1 --topk 10 --intra_loss 2

for benchmark:

for example: book -> dvd

	1. generate last 10-layer features:
		python generate.xlmr.topk.py ./data/amazon/book/raw  
		python generate.xlmr.topk.py ./data/amazon/book/train  
		python generate.xlmr.topk.py ./data/amazon/dvd/raw 
		python generate.xlmr.topk.py ./data/amazon/dvd/train 
		python generate.xlmr.topk.py ./data/amazon/dvd/test
		python generate.xlmr.topk.py ./data/amazon/book/test

	2. run CFd.bench.py
		python CFd.bench.py --raw_train_path  ./data/amazon/book/raw  ./data/amazon/book/train  ./data/amazon/dvd/raw ./data/amazon/dvd/train  --mode train_MI --n_epochs 20 --lr 0.0005 --text_lr 0.0005 --hidden_size 256 --batch_size 200 --text_batch_size 50 --neg_n 10 --nclass 2  --train_path ./data/amazon/book/train  --test_path ./data/amazon/dvd/test --valid_path ./data/amazon/book/test --save_mi_path path_to_save  --random_lstm 1  --lamda 5000 --mi_lamda_s 0 --mi_lamda_t 1  --pseudo_t_lamda 1  --cluster_lamda 1 --topk 10 --intra_loss 2

The commonds will evaluate the model performance with 5 random runs and the results include the methods of p, kl, mmd, adv, mi, mi+c, mi+p, c+p, mi+c+p. 

### To get the results for xlmr-1 and xlmr-10
for monoAmazon:

	book -> music:
	-for xlmr-1:
		python CFd.monoAmazon.py  --raw_train_path   ./data/small/book/set1_text.txt  ./data/small/music/set1_text.txt  --mode train --n_epochs 20 --lr 0.0001 --text_lr 0.0001 --hidden_size 256  --batch_size 50 --text_batch_size 100 --neg_n 10 --nclass 3  --train_path  ./data/small/book/train.txt --valid_path  ./data/small/book/test.txt  --test_path  ./data/small/music/all.txt --save_mi_path path_to_save --random_lstm 1 --lamda 0 --mi_lamda_s 0 --mi_lamda_t 1  --pseudo_t_lamda 1  --cluster_lamda 1 --topk 1 
	-for xlmr-10:  
		change --topk to 10

for multiAmazon:

	en, books -> fr, books  
	-for xlmr-1:
		python CFd.multiAmazon.py --raw_train_path ./data/sampled_raw_data/6000_fr_books_raw.txt ./data/annotated_data/fr/books/train.txt  --mode train --n_epochs 20 --lr 0.0005 --text_lr 0.0005 --hidden_size 256 --batch_size 200 --text_batch_size 50 --neg_n 10 --nclass 2  --train_path ./data/annotated_data/en/books/train.txt --test_path ./data/annotated_data/fr/books/test.txt --valid_path ./data/annotated_data/en/books/test.txt --save_mi_path path_to_save --random_lstm 1 --lamda 0 --mi_lamda_s 0 --mi_lamda_t 1  --pseudo_t_lamda 1  --cluster_lamda 1 --topk 1 
	-for xlmr-10:
		change --topk to 10

for benchmark

	book -> dvd
	-for xlmr-1:	
		python CFd.bench.py --raw_train_path  ./data/amazon/book/raw  ./data/amazon/book/train ./data/amazon/dvd/raw ./data/amazon/dvd/train --mode train --n_epochs 20 --lr 0.0005 --text_lr 0.0005 --hidden_size 256 --batch_size 200 --text_batch_size 50 --neg_n 10 --nclass 2  --train_path ./data/amazon/book/train --test_path ./data/amazon/dvd/test --valid_path ./data/amazon/book/test --save_mi_path path_to_save --random_lstm 1 --inter_time 3  --lamda 5000 --mi_lamda_s 0 --mi_lamda_t 1  --pseudo_t_lamda 1  --cluster_lamda 1 --topk 1 --intra_loss 1
	-for xlmr-10:
		change --topk to 10


## Please cite our work if you use this code:
```
 @inproceedings{ye2020feature,
  title={Feature Adaptation of Pre-Trained Language Models across Languages and Domains with Robust Self-Training},
  author={Ye, Hai and Tan, Qingyu and He, Ruidan and Li, Juntao and Ng, Hwee Tou and Bing, Lidong},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages={7386--7399},
  year={2020}
}
```
