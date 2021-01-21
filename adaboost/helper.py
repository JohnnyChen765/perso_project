import pandas

features_bank = {
    'height' : 100,
    'pointy_ear' : False,
    'life_expectancy' : 100,
    'violence' : False,
    'strength' : 10,
    'agility' : 10
}

def orc_generator(n_individus):
    list_individu = []

    features = features_bank.copy()
    features['height'] += 100
    features['violence'] = True
    features['strength'] += 10
    features['agility'] -= 5
    features['race'] = 'orc'

    for i in range (1, n_individus):
        list_individu.append(features)
    
    return pandas.DataFrame(list_individu)

def elf_generator(n_individus):
    list_individu = []

    features = features_bank.copy()
    features['pointy_ear'] = True
    features['life_expectancy'] = 200
    features['race'] = 'elf'

    for i in range (1, n_individus):
        list_individu.append(features)
    
    return pandas.DataFrame(list_individu)

def human_generator(n_individus):
    list_individu = []

    features = features_bank.copy()
    features['race'] = 'human'

    for i in range (1, n_individus):
        list_individu.append(features)
    
    return pandas.DataFrame(list_individu)


