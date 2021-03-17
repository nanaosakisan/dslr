def scale(dataset):
    data_scale = dataset.copy()
    min_data = min(min(data_scale, key=min))
    max_data = max(max(data_scale, key=max))
    data_scale = [[((x - min_data) / (max_data - min_data)) for x in sub] for sub in data_scale]
    return data_scale, min_data, max_data

def encode_house(dataset, index1, index2):
    encode_house = []
    dict_house = {"Ravenclaw": 0, "Slytherin":1, "Gryffindor":2, "Hufflepuff":3}
    for row in dataset[1:]:
        if row[index1] is not None and row[index2] is not None:
            push = dict_house.get(row[1])
            encode_house.append(push)
    return encode_house

def get_feature(dataset, index1, index2):
    feature = []
    for row in dataset[1:]:
        if row[index1] is not None and row[index2] is not None:
            push = [row[index1], row[index2]]
            feature.append(push)
    return feature

def preprocess(dataset, feature1, feature2):
    # preprocess qui va garder uniquement features choisie pour train, encoder la maison
    # et scale les features numérique

    sub_data = [["House", "Feature1", "Feature2"]]
    house = encode_house(dataset, feature1, feature2)
    feature = get_feature(dataset, feature1, feature2)
    feature, min_, max_ = scale(feature)
    for i, row in enumerate(house):
        push = [row, feature[i][0], feature[i][1]]
        sub_data.append(push)
    return sub_data