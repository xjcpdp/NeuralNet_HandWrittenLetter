from typing import Tuple
import numpy as np
import sys

def sigmoid(x:np.ndarray) -> np.ndarray:
    return 1/(1+(np.exp(-x)))

def soft_max(x: np.ndarray) -> np.ndarray:
    return np.exp(x) / np.sum(np.exp(x))

# something went wrong here
def cross_entropy(y_hat: np.ndarray, y_true: np.ndarray) -> float:
    y_true = y_true.reshape(y_true.shape[0],1)
    return -1 * y_true.transpose().dot(np.log(y_hat))

def train(num_epoch: int) -> str:
    train_str = ''
    global train_x, train_true, alpha, beta, train_label, test_in

    for j in range(num_epoch):
        for i in range(train_x.shape[0]):
            row = train_x[i]
            y_true = train_true[i]
            y_true = y_true.reshape(y_true.shape[0],1)
            row = row.reshape(row.shape[0],1)
            y_hat = forward(row, y_true)
            g_alpha, g_beta = backward(row, y_hat, y_true)
            # update via SGD
            alpha = alpha - learning_rate * g_alpha
            beta = beta - learning_rate * g_beta

        entropy = []
        for i in range(train_x.shape[0]):
            row = train_x[i]
            y_true = train_true[i]
            y_true = y_true.reshape(y_true.shape[0],1)
            row = row.reshape(row.shape[0],1)
            y_hat = forward(row, y_true)
            J = float(cross_entropy(y_hat, y_true))
            entropy.append(J)
        train_str += ('epoch=%d '%(j+1))
        train_str += 'crossentropy(train): %f\n'% (sum(entropy)/len(entropy))
        train_str += ('epoch=%d '%(j+1))
        train_str += 'crossentropy(test): %f\n'% test_entropy(test_in)

    # generate label and error rate
    result_predicted = []

    for i in range(train_x.shape[0]):
        train_lab = train_true[i]
        train_input = train_x[i]
        train_lab = train_lab.reshape(train_lab.shape[0],1)
        train_input = train_input.reshape(train_input.shape[0],1)
        train_y_hat = forward(train_input, train_lab)
        result_predicted.append(np.argmax(train_y_hat))

    error_rate_test = check_error(result_predicted, train_label)
    train_str += 'error(train): %f'%error_rate_test
    return train_str, result_predicted

def test_entropy(test_in: str) -> float:
    entropy = []
    test_data, labels = read_train(test_in)
    test_labels = one_hot_encode(labels)
    for i in range(test_data.shape[0]):
        test_label = test_labels[i]
        test_input = test_data[i]
        test_label = test_label.reshape(test_label.shape[0], 1)
        test_input = test_input.reshape(test_input.shape[0], 1)
        # print(test_label)
        # print(test_input)
        test_y_hat = forward(test_input, test_label)
        J = float(cross_entropy(test_y_hat,test_label))
        entropy.append(J)
    return sum(entropy)/len(entropy)

def test(test_in: str) -> float:
    result_predicted = []
    test_data, labels = read_train(test_in)
    test_labels = one_hot_encode(labels)
    entropy = []
    for i in range(test_data.shape[0]):
        test_label = test_labels[i]
        test_input = test_data[i]
        test_label = test_label.reshape(test_label.shape[0], 1)
        test_input = test_input.reshape(test_input.shape[0], 1)
        # print(test_label)
        # print(test_input)
        test_y_hat = forward(test_input, test_label)
        J = cross_entropy(test_y_hat,test_label)
        entropy.append(J)
        result_predicted.append(np.argmax(test_y_hat))
    error_rate_test = check_error(result_predicted, labels)
    return error_rate_test, result_predicted

def check_error(predicted_list, true):
    count = 0
    for i in range(len(predicted_list)):
        predicted = predicted_list[i]
        true_label = true[i]
        if predicted != true_label:
            count += 1
    return count/len(predicted_list)


def backward(row: np.ndarray, y_hat: np.ndarray, y_true: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    global learning_rate,alpha, beta, b, z, a
    dl_db = y_hat - y_true
    # print(dl_db)
    temp = beta[:,:-1]
    dl_dz = dl_db.transpose().dot(temp)
    # print(dl_dz)
    dl_da = dl_dz * (((np.exp(-a))/((1+np.exp(-a))**2)).transpose())
    print(dl_da.shape)
    # print(dl_da.shape)
    # print(row)
    # print(row.shape)
    dl_dalpha = (dl_da.transpose() * row.transpose())
    # print(dl_dalpha.shape)
    # print(dl_dalpha)
    dl_dbeta = dl_db.dot(z.transpose())
    # print(dl_dbeta.transpose())

    return dl_dalpha, dl_dbeta

def forward(row: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    global alpha, beta, b, z, a, train_true, j, entropy
    a = alpha.dot(row)
    z = sigmoid(a)
    z = np.vstack((z, np.ones((1,1))))
    b = beta.dot(z)
    y_hat = soft_max(b)
    j = float(cross_entropy(y_hat, y_true))
    return y_hat

def one_hot_encode(label: np.ndarray) -> np.ndarray:
    re = np.zeros((label.shape[0],10))
    for i in range(label.shape[0]):
        idx = int(label[i][0])
        re[i,idx] = 1
    return re

def read_train(train_in: str) -> Tuple[np.ndarray, np.ndarray]:
    train_data = np.genfromtxt(train_in, delimiter=',')
    labels = train_data[:,[0]]
    train_data = train_data[:,1:]
    shape = train_data.shape
    bias = np.ones((shape[0],1))
    return np.hstack((train_data,bias)), labels

def write_output(out:str, out_str: str):
    fl_out = open(out,'w')
    fl_out.write(out_str)
    fl_out.close()
    return

# Main function starts here
if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)

    train_in = sys.argv[1]
    test_in = sys.argv[2]
    train_out = sys.argv[3]
    test_out = sys.argv[4]
    metric_out = sys.argv[5]
    num_epoch = int(sys.argv[6])
    hidden_unites = int(sys.argv[7])
    init_flag = int(sys.argv[8])
    learning_rate = float(sys.argv[9])

    # initialize x (input vector)
    train_x, train_label = read_train(train_in)
    train_true = one_hot_encode(train_label)
    # entropy = []
    # initialize alpha and beta
    # init_flag = 1
    alpha = np.zeros((hidden_unites, train_x.shape[1]))
    beta = np.zeros((10,hidden_unites+1))
    if init_flag == 1:
        alpha = np.random.uniform(low= -0.1, high= 0.1, size= (hidden_unites,train_x.shape[1]-1))
        alpha = np.hstack((alpha, np.zeros((hidden_unites,1))))
        beta = np.random.uniform(low = -0.1, high= 0.1, size = (10, hidden_unites))
        beta = np.hstack((beta, np.zeros((10,1))))

    train_str, train_predicted = train(num_epoch)
    error, test_predicted = test(test_in)
    train_str += '\nerror(test): %f' % error
    print(train_str)
    write_output(metric_out, train_str)
    train_label_str = ''
    for i in train_predicted:
        train_label_str += str(i)+'\n'
    test_label_str = ''
    for i in test_predicted:
        test_label_str += str(i)+'\n'
    write_output(train_out, train_label_str)
    write_output(test_out, test_label_str)