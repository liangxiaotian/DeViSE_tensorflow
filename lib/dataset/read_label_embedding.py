import numpy as np

# 相似度越高,值越大
def get_labels_embedding(embedding_file):
    label_embedding_dictory = dict()
    lable_embedding = np.load(embedding_file)
    lable_embedding_reshape = np.reshape(lable_embedding, (100, 300))

    return lable_embedding_reshape

if __name__ == '__main__':
    dir = get_labels_embedding('../../data/label_embedding.npy')
    x = []
    for i in range(3):
        x.append(dir[i])
    print(x)
