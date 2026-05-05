from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def evaluate_model(y_test, y_pred, title="Modelo"):
    print(f"\n===== {title} =====")
    print(classification_report(y_test, y_pred))
    print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))


def show_prediction(image, prediction):
    plt.imshow(image.reshape(28,28), cmap='gray')
    plt.title(f"Predicción: {prediction}")
    plt.axis('off')
    plt.show()