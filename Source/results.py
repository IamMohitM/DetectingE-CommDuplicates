import pickle, json

with open('Final.pickle', 'rb') as f:
    results_dict = pickle.load(f)

with open(r'duplicates.pickle', 'rb') as f:
    image_dict = pickle.load(f)

# for key in results_dict.keys():
#     results_dict[key] = [[val] for val in results_dict[key]]


for key in image_dict.keys():
    try:
        # results_dict[key] += list(set(image_dict[key]))
        results_dict[key] += [[val] for val in image_dict[key]]

    except KeyError:
        # results_dict[key] = list(set(image_dict[key]))
        results_dict[key] = [[val] for val in image_dict[key]]

with open(r'..\Submission Files\Final_Results_with_required_format.json', 'w') as fp:
    json.dump(results_dict, fp)
