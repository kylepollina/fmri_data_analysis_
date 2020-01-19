
def confusion_matrix(x, y, predictions, correct_labels):
    confusion_matrix = np.zeros((12, 12))

    for i in range(len(predictions)):
        predicted_label = predictions[i]
        correct_label = self.y_test[i]

        confusion_matrix[predicted_label][correct_label] += 1

    return confusion_matrix

