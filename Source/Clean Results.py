import pickle

result_dict = {}
with open('LSH_Similar_images.txt', 'r') as f:
    text = f.read()
    l = eval(text)
    for dictionary in l:
        for key, value in dictionary.items():
            result_dict[key] = list(set(value))

    print(result_dict)
    with open(r'..\Submission Files\Final.pickle', 'wb') as fr:
        pickle.dump(result_dict, fr)