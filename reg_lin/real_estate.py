# On importe les librairies dont on aura besoin pour ce tp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import reg_lin

# On charge le dataset
house_data = pd.read_csv('house.csv')

# On affiche le nuage de points dont on dispose
def show():
    plt.plot(house_data['surface'], house_data['loyer'], 'ro', markersize=4)
    plt.grid()
    plt.show()
    return 0

house_data = house_data[house_data['loyer'] < 10000]

# On décompose le dataset et on le transforme en matrices pour pouvoir effectuer notre calcul
X = np.matrix([np.ones(house_data.shape[0]), house_data['surface']]).T
y = np.matrix(house_data['loyer']).T

# On effectue le calcul exact du paramètre theta
theta = reg_lin.custom_reg_lin(X, y)

print(theta)

def show_regression():
    plt.xlabel('Surface')
    plt.ylabel('Loyer')

    plt.plot(house_data['surface'], house_data['loyer'], 'ro', markersize=4)
    # On affiche la droite entre 0 et 250
    plt.plot([0,250], [theta.item(0),theta.item(0) + 250 * theta.item(1)], linestyle='--', c='#000000')
    plt.show()
    return 0

show_regression()