# DeViSE_tensorflow
[Implement DeViSE: A Deep Visual-Semantic Embedding Model][https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/41473.pdf]
## 1.Get word2vec from pretrain googlenews
  Down pretrain model [GoogleNews-vectors-negative300.bin.gz][https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit]  
  Get cifar-100 word2vec  
  python tools/get_word2vec.py
  
## 2.pretrain the AlexNet on cifar-100
  python tools/train.py
  
## 3.Fine-tune the projection layer M
  python tools/DeViSE_train.py
  
## 4.Results
| model 	       |   embedding |  acc    |
| -----------------| :----------:|:-------:|
| softmax_baseline |    300		 |  0.5704 |
| DeViSE           |    300      |  0.4797 |


