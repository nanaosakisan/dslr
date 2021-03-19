from datetime import datetime

def is_number(s):
    if s is not None:
        try:
            float(s)
        except ValueError:
            return False
    return True

def check_features(content) :
    if len(content) != 19 :
        return 0, "Erreur dans le nombre de features. Nombre de features attendu : 19"
    features = ["Index", "Hogwarts House", "First Name", "Last Name", \
         "Birthday", "Best Hand", "Arithmancy", "Astronomy", "Herbology", \
        "Defense Against the Dark Arts", "Divination", "Muggle Studies", \
        "Ancient Runes", "History of Magic", "Transfiguration", \
        "Potions", "Care of Magical Creatures", "Charms", "Flying"]

    for i, obj in enumerate(content):
        if obj != features[i] :
            return 0, "Erreur dans le nom de feature à la ligne : " + str(i) + ". Feature \
            attendu : " + features[i]
    return 1, ""

def check_house(dataset):
    house = ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]
    for i, obj in enumerate(dataset[1:]):
        if obj[1] not in house :
            return 0, "Maison inconnue à la ligne : " + str(i) + """. Feature attendu : \
            ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]."""
    return 1, ""

def check_birthday(dataset):
    for i, row in enumerate(dataset[1:]):
        try:
            datetime.strptime(row[4], "%Y-%m-%d")
        except ValueError:
            return 0, "Le format de la date est incorrect à la ligne : " + str(i) + ". \
            Format attendu : YYYY-MM-DD"
    return 1, ""

def check_hand(dataset):
    for i, row in enumerate(dataset[1:]):
        if row[5] != "Left" and row[5] != "Right" :
            return 0, "Main inconnue à la ligne : " + str(i) + ". Main attendue : \
            Left ou Right"
    return 1, ""

def check_grade(dataset) :
    for i, row in enumerate(dataset[1:]):
        for j, col in enumerate(row[6:]):
            if len(col) == 0:
                dataset[i+1][j+6] = None
                col = None
            if is_number(col) is False :
                return 0, "Note inconnue à la ligne : " + str(i) + ", colonne : " + str(j+6) +\
                ". Valeur numérique attendue." + col
            else:
                if col is not None:
                    dataset[i+1][j+6] = float(col)
    return 1, "", dataset


def parse_train(filename) :
    lines = filename.read().decode("utf-8").split("\n")
    dataset = []

    for i, line in enumerate(lines):
        content = line.split(",")
        if i == 0 :
            res, error = check_features(content)
            if res == 0:
                return 0, error, dataset 
        if len(content) == 19 :
            dataset.append(content)
    filename.close()
    res, error = check_house(dataset)
    res, error = check_birthday(dataset)
    res, error = check_hand(dataset)
    res, error, dataset = check_grade(dataset)
    if res == 0:
        return 0, error, dataset
    return 1, "", dataset

def check_house_pred(dataset):
    house = ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]
    for obj in dataset[1:5] :
        st.write(obj)
        if obj[1] not in house :
            return 0, "Maison inconnue à la ligne : " + str(i) + """. Feature attendu : \
            ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]."""
    return 1, ""

def parse_predict(filename) :
    lines = filename.read().decode("utf-8").split("\n")
    dataset = []

    for i, line in enumerate(lines):
        content = line.split(",")
        if i == 0 :
            res, error = check_features(content)
            if res == 0:
                return 0, error, dataset 
        if len(content) == 19 :
            dataset.append(content)
    filename.close()
    res, error = check_house_pred(dataset)
    res, error = check_birthday(dataset)
    res, error = check_hand(dataset)
    res, error, dataset = check_grade(dataset)
    if res == 0:
        return 0, error, dataset
    return 1, "", dataset