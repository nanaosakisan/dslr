def scale(dataset, min_, max_):
    data_scale = dataset.copy()
    data_scale = [[((x - min_) / (max_ - min_)) for x in sub] for sub in data_scale]
    return data_scale

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

def preprocess_predict(dataset, feature1, feature2, min_, max_):
    # preprocess qui va garder uniquement features choisie pour train, encoder la maison
    # et scale les features numérique

    sub_data = [["House", "Feature1", "Feature2"]]
    house = encode_house(dataset, feature1, feature2)
    feature = get_feature(dataset, feature1, feature2)
    feature = scale(feature, min_, max_)
    for i, row in enumerate(house):
        push = [row, feature[i][0], feature[i][1]]
        sub_data.append(push)
    return sub_data
