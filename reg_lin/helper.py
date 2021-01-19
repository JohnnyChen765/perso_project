import numpy as np


def newline():
    print("\n")


def get_MSE(error_hat):
    n = error_hat.shape[0]
    norme_carre_error_hat = error_hat.T.dot(error_hat)
    mse = norme_carre_error_hat / n
    rmse = np.sqrt(mse)

    newline()
    print("Le MSE obtenu sur les prédictions")
    print(mse)

    newline()
    print("Le RMSE obtenu sur les prédictions")
    print(np.sqrt(norme_carre_error_hat / n))
    return (mse, rmse)


def get_MSER(error_hat, y):
    n = error_hat.shape[0]
    error_hat_relatif = error_hat / y
    norme_carre_error_hat_relatif = error_hat_relatif.T.dot(error_hat_relatif)
    MSE_relatif = norme_carre_error_hat_relatif / n

    newline()
    print(
        "Le MSE_relatif obtenu sur les prédictions"
    )  # a priori, peu utile, car le carré écrase les valeurs, car elles sont < 1.
    print(MSE_relatif)
    return MSE_relatif


def get_MAER(error_hat, y):
    n = error_hat.shape[0]
    error_hat_relatif_abs = np.abs(error_hat / y)
    norme_carre_error_hat_relatif_abs = error_hat_relatif_abs.T.dot(np.ones(n))
    MAE_relatif = norme_carre_error_hat_relatif_abs / n
    newline()
    print(
        "Le MAE_relatif_ obtenu sur les prédictions"
    )  # le soucis est que cette métrique n'accentue pas les grosses erreurs. Le fait d'être en relatif empêche l'accentuation des erreurs.
    print(MAE_relatif)
