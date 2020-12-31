import numpy as np  # 1.18.0
import pandas as pd  # 0.25.3
import matplotlib.pyplot as plt  # 3.1.2
# Python 3.6.9


def sigmoid(x):
    return 1 / (1 + np.exp(-0.005*x))


def sigmoid_derivative(x):
    return 0.005 * x * (1 - x)


def read_and_divide_into_train_and_test(csv_file):
    df = pd.read_csv(csv_file, na_values=["?"])
    df.drop(["Code_number"], 1, inplace=True)
    df = df.fillna(df.mean())
    # Show correlation graph
    Y_data = df['Class']
    X_data = df.copy()
    X_data.drop(['Class'], axis=1, inplace=True)
    indices = list(range(X_data.shape[0]))
    num_training_indices = int(0.8 * X_data.shape[0])
    np.random.shuffle(indices)
    train_indices = indices[:num_training_indices]
    test_indices = indices[num_training_indices:]
    # split the actual data
    training_inputs, training_labels = X_data.iloc[train_indices], Y_data.iloc[train_indices]
    test_inputs, test_labels = X_data.iloc[test_indices], Y_data.iloc[test_indices]
    df.drop(["Class"], 1, inplace=True)
    length = df.corr().shape
    plt.matshow(df.corr())
    plt.xticks(range(len(df.columns)), df.columns, rotation = 90)
    plt.yticks(range(len(df.columns)), df.columns)
    plt.colorbar()

    for i in range(length[0]):
        for j in range(length[1]):
            plt.text(i, j, round(df.corr().iloc[i, j],2), ha="center", va="center", color="r")
    plt.show()
    return training_inputs, training_labels, test_inputs, test_labels


def run_on_test_set(test_inputs, test_labels, weights):
    tp = 0
    test_outputs = test_inputs.dot(weights)
    test_outputs = sigmoid(test_outputs)
    for i in range(len(test_labels)):
        if float(test_labels.iloc[i]) == round(test_outputs.iloc[i]):
            tp += 1
    accuracy = tp / len(test_labels)
    return accuracy


def plot_loss_accuracy(accuracy_array, loss_array):
    plt.subplot(2, 1, 1)
    plt.plot(loss_array, '-')
    plt.title('Accuracy and Loss Plot')
    plt.ylabel('Loss')
    plt.subplot(2, 1, 2)
    plt.plot(accuracy_array, '-')
    plt.xlabel('# Epochs')
    plt.ylabel('Accuracy')
    plt.show()


def main():
    csv_file = './breast-cancer-wisconsin.csv'
    iteration_count = 2500
    np.random.seed(1)
    weights = 2 * np.random.random((9,)) - 1
    accuracy_array = []
    loss_array = []
    training_inputs, training_labels, test_inputs, test_labels = read_and_divide_into_train_and_test(csv_file)

    for iteration in range(iteration_count):
        outputs = training_inputs.dot(weights)
        outputs = sigmoid(outputs)
        loss = training_labels - outputs
        tuning = loss*sigmoid_derivative(outputs)
        update_val = np.dot(training_inputs.T, tuning)
        weights += update_val
        accuracy_array.append(run_on_test_set(test_inputs, test_labels, weights))
        loss_array.append(-loss.mean())
    plot_loss_accuracy(accuracy_array, loss_array)


if __name__ == '__main__':
    main()
