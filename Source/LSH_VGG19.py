import cv2
import string
import urllib.request
import numpy as np
import pandas as pd
from tqdm import tqdm

from datasketch import MinHash, MinHashLSH
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from nltk import FreqDist, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

model = VGG19(weights='imagenet', include_top=False)
table = str.maketrans(dict.fromkeys(string.punctuation))
image_feature = {}
LSH_RESULTS = 'LSH File.txt'
LSH_IMAGE_RESULTS = 'Lsh_Similar_images.txt'
PROCESSED_GROUPS = 'Processed groups.txt'

#return list of k-grams
def ngrams(text, n=3):
    n_grams = zip(*[text[i:] for i in range(n)])
    return [''.join(ngram) for ngram in n_grams]

#Download image from the url or return feature of the image if already downloaded
#If there is an error, a black image will be returned
def download_image(img_url):
    try:
        return image_feature[img_url]
    except KeyError:
        try:
            url_response = urllib.request.urlopen(img_url, timeout=30)
            img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
            img = cv2.imdecode(img_array, -1)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
			img = np.dstack((img, img, img))
            img = cv2.resize(img, (224, 224))
            return img
        except Exception:
            return np.zeros((224, 224, 3))

#Extract features from an image using VGG19
def extract_features(img_url, img):
    try:
        return image_feature[img_url]
    except KeyError:
        cv2.imwrite(img_url, img)
        img_data = np.expand_dims(img, axis=0)
        img_data = preprocess_input(img_data)
        feature = model.predict(img_data)
        image_feature[img_url] = feature
    return feature

#Compare images and return product ids of a similar images 
def compare_images(image_url, ids, other_urls):
    image = download_image(image_url)
    other_images = [(url, download_image(url)) for url in other_urls]
    main_img = np.expand_dims(extract_features(image_url, image).flatten(), axis=0)
    other_images = [extract_features(url, img).flatten() for url, img in other_images]
    if not other_images:
        return []
    cosine_mat = cosine_similarity(main_img, other_images)
    ids = np.array(ids)
    return ids[np.argwhere(cosine_mat > 0.45)].flatten().tolist()

#Locality Sensitive Hashing
def apply_lsh(group, col):
    lsh = MinHashLSH(threshold=0.9, num_perm=256)
    minhashes = {}
    for idx, text in group[col].iteritems():
        minhash = MinHash(num_perm=256)
        for d in ngrams(text, 3):
            minhash.update("".join(d).encode('utf-8'))
        index = group.loc[idx, 'productId']
        lsh.insert(key=index, minhash=minhash)
        minhashes[index] = minhash
    return lsh, minhashes

#Return frequency of each word in a column
def word_frequency(column):
    word_list = []
    for row in column:
        word_list += word_tokenize(row)
    return FreqDist(word_list)


def remove_freq_words(text, frequent_words):
    edited_text = ' '.join([word for word in text.split() if word.lower() not in frequent_words])
    return edited_text

#Function applied to each gorup to find duplicates
def find_all_duplicates(group):
    object_cols = ['key_specs_text', 'description', 'title']
    similar_images = {}
    duplicate_dict = {}

    print(f'{group.name}: {group.shape}')

    for col in object_cols:
        group[col] = group[col].fillna('').str.lower().str.translate(table)

	#Concatenating the text columns
    group['full_text'] = group['key_specs_text'].astype(str) + ' ' + group['description'].astype(str) + ' ' + group[
        'title'].astype(str)
    group['imageUrl'] = group['imageUrl'].fillna('')

    word_freq = word_frequency(group['full_text'])
    frequent_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
    group['full_text'] = group['full_text'].apply(lambda text: remove_freq_words(text, frequent_words))
    lsh, minhashes = apply_lsh(group, 'full_text')
	
	#Filtering out ids resulting from LSH for image comparision
    for key in minhashes.keys():
        result = lsh.query(minhashes[key])
        result = list(set([value for value in result if key != value]))
        duplicate_dict[key] = result
        if not result:
            continue
        key_url = group.loc[group['productId'] == key, 'imageUrl'].values[0]
        result_url = group.loc[group['productId'].isin(result), 'imageUrl'].values.tolist()
        similar_images[key] = list(set(compare_images(key_url, result, result_url)))

	#Results will be extracted from these files
    with open(LSH_IMAGE_RESULTS, 'a+') as f:
        print(similar_images, file=f)
    with open('LSH_RESULTS', 'a+') as f:
        print(duplicate_dict, file=f)
    with open(PROCESSED_GROUPS, 'a+') as p:
        print(group.name, file=p)

    image_features = {}
    print('Done')


if __name__ == "__main__":
    df = pd.read_csv(r'Final Data.csv')
    df = df.drop_duplicates(subset=['productId'])
	
	#Just to avoid reprocessing of procced gorups
    with open('Processed groups.txt', 'r') as f:
        processed_groups = []
        for line in f.readlines():
            processed_groups.append(eval(line))
    try:
        processed_brands, processed_cat1 = zip(*processed_groups)

        df = df[~((df['productBrand'].isin(processed_brands)) &
                  (df['sub_category1'].isin(processed_cat1)))]
    except ValueError:
        pass
    tqdm.pandas()
    print(df.shape)

    group_cols = ['productBrand', 'sub_category1']

    df = df.groupby(group_cols).filter(lambda group: len(group) > 1)
    grouped = df.groupby(group_cols)
    grouped.progress_apply(find_all_duplicates)
