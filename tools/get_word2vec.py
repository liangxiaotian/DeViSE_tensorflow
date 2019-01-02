import gensim
import numpy as np

# load pre-train word2vec embedding
model = gensim.models.KeyedVectors.load_word2vec_format(
    '/home/xiaotian/Datasets/Pretrain_word2vec/GoogleNews-vectors-negative300.bin', binary=True)
x = list(model.vocab.keys())[:10]
print(x)


# cifar 100 dataset labels
def get_label_embedding():
    CIFAR100_LABELS_LIST = [
        'apple', 'fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
        'worm'
    ]

    maxtrix = np.zeros((100, 300))
    maxtrix = np.reshape(maxtrix, (100, 300))
    # save label embedding as label_embedding.npy
    for i in range(100):
        string = model[str(CIFAR100_LABELS_LIST[i])]
        maxtrix[i] = string
    maxtrix = np.reshape(maxtrix, (100 * 300))
    np.save('../data/label_embedding.npy', maxtrix)


# Zero shot label embedding
def get_zero_shot_label_embedding():
    return NotImplementedError

if __name__ == "__main__":
    get_label_embedding()
    # get_zero_shot_label_embedding()
