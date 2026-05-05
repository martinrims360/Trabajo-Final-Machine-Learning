import numpy as np

from preprocessing import load_data, flatten_data
from cnn_model import create_cnn
from traditional_models import train_knn, train_svm
from utils import evaluate_model, show_prediction

def main():

    # 1. Cargar datos
    X_train, X_test, y_train, y_test = load_data()

    # 2. Datos planos (para KNN y SVM)
    X_train_flat, X_test_flat = flatten_data(X_train, X_test)

    # ==========================
    # 🔷 CNN
    # ==========================
    print("\nEntrenando CNN...")
    cnn = create_cnn()
    cnn.fit(X_train, y_train, epochs=3, validation_split=0.2)

    cnn_preds = cnn.predict(X_test)
    cnn_preds = np.argmax(cnn_preds, axis=1)

    evaluate_model(y_test, cnn_preds, "CNN")

    # ==========================
    # 🔷 KNN
    # ==========================
    print("\nEntrenando KNN...")
    knn = train_knn(X_train_flat, y_train)
    knn_preds = knn.predict(X_test_flat)

    evaluate_model(y_test, knn_preds, "KNN")

    # ==========================
    # 🔷 SVM
    # ==========================
    print("\nEntrenando SVM...")
    svm = train_svm(X_train_flat, y_train)
    svm_preds = svm.predict(X_test_flat)

    evaluate_model(y_test, svm_preds, "SVM")

    # ==========================
    # 🔷 Visualización
    # ==========================
    print("\nMostrando ejemplo...")
    show_prediction(X_test[0], cnn_preds[0])


if __name__ == "__main__":
    main()