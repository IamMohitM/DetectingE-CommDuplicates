import numpy as np
import string
from datasketch import MinHash, MinHashLSH
from keras.applications.vgg19 import VGG19
from sklearn.metrics.pairwise import cosine_similarity

from LSH_VGG19 import download_image, ngrams, extract_features

table = str.maketrans(dict.fromkeys(string.punctuation))


def compare_products(product1, product2):
    """Checks if the two given series/dicts(strictly) belong to the same product
    The keys should strictly be followed as per the dataset.
    """

    if product1['imageUrl'] == product2['imageUrl']:
        print('Yes')
        return
    text_cols = ['key_specs_text', 'description', 'title']
    id1 = product1['productId']

    check_image = False
    product1['full_text'] = ''
    product2['full_text'] = ''

    for col in text_cols:
        product1['full_text'] += ' ' + product1[col].translate(table)
        product2['full_text'] += ' ' + product2[col].translate(table)

    m1 = MinHash(num_perm=258)
    m2 = MinHash(num_perm=258)

    for d in ngrams(product1['full_text'], 3):
        m1.update(d.encode('utf-8'))
    for d in ngrams(product2['full_text'], 3):
        m2.update(d.encode('full_text'))

    lsh = MinHashLSH(threshold=0.9, num_perm=256)
    lsh.insert(id1, m1)
    result = lsh.query(m2)
    if id1 in result:
        print('Similar Text')
        check_image = True
    if not check_image:
        print('No')
    else:
        print(product1['imageUrl'])
        print(product2['imageUrl'])

        img1 = download_image(product1['imageUrl'])
        img2 = download_image(product2['imageUrl'])

        img1 = np.expand_dims(extract_features(product1['imageUrl'], img1), axis=0)
        img2 = np.expand_dims(extract_features(product2['imageUrl'], img2), axis=0)

        cosine_mat = cosine_similarity(img1, img2)
        if cosine_mat > 0.5:
            print('Yes')
        else:
            print('No')

