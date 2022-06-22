import json

def load_params(path = "settings/settings.json"):
    """

    :param path: path to file that contains all the settings.
    :return: Dictionnary with all necessary settings.
    """
    with open(path) as json_file:
        data = json.load(json_file)

    return data


if __name__ == "__main__":
    #For bug correction purpose.
    data = load_params()
    print(data)