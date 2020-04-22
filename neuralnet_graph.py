__author__ = 'Yihan Qiu, yihanq@andrew.cmu.edu, AndreID = yihanq'

import sys
import numpy as np
import math
import matplotlib.pyplot as plt

class neuralNetwork:
    def __init__(self, train_feature, train_label, test_feature, test_label, init_flag, learning_rate, hidden_units,
                 epochs):
        self.train_feature = train_feature
        self.train_label = train_label
        self.test_feature = test_feature
        self.test_label = test_label
        self.alpha, self.beta = self.init_param(init_flag)
        self.lr = learning_rate
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.train_mean_ce = []
        self.test_mean_ce = []

    def linearForward(self, x, alpha):
        x = np.hstack((np.ones(1), x))
        return alpha.dot(x)

    def linearBackward(self, x, alpha, a, grad_a):
        x = np.hstack((np.ones(1), x))
        x = x[:, np.newaxis]
        grad_alpha = grad_a[:, np.newaxis].dot(np.transpose(x))
        grad_x = (alpha.T).dot(grad_a)[1:]
        return grad_alpha, grad_x

    def sigmoidForward(self, a):
        return 1 / (1 + np.exp(-a))

    def sigmoidBackward(self, a, z, grad_z):
        grad_a = grad_z * z * (1 - z)
        return grad_a

    def softmaxForward(self, b):
        sum = 0
        for i in b:
            sum += math.exp(i)
        return np.exp(b) / sum

    # def softmaxBackward(self, b, y_pred, grad_y_pred):
    #     temp = np.diag(y_pred) - y_pred.dot(y_pred.T)
    #     grad_b = (grad_y_pred.T).dot(temp)
    #     return grad_b

    def softmaxCrossentropyBackward(self, target, pred, grad_J):
        return pred - target

    def softmaxBackward(self, b, y_pred, grad_y_pred):
        temp = np.diag(y_pred) - y_pred.dot(y_pred.T)
        grad_b = temp.dot(grad_y_pred)
        return grad_b

    def crossEntropyForward(self, target, predict):
        return -1 * ((target.T).dot(np.log(predict)))

    def crossEntropyBackward(self, target, y_pred, J, grad_J):
        grad_y_pred = -1 * grad_J * np.divide(target, y_pred)
        return grad_y_pred

    def NNForward(self, train_feature, train_label, alpha, beta):
        a = self.linearForward(train_feature, alpha)
        # print('First layer output: {}'.format(a))
        z = self.sigmoidForward(a)
        # print('Hidden layer output{}'.format(z))
        b = self.linearForward(z, beta)
        # print('Second layer output: {}'.format(b))
        y_pred = self.softmaxForward(b)
        # print('Final output: {}'.format(y_pred))
        J = self.crossEntropyForward(train_label, y_pred)
        # print('CrossEntropyLoss: {}'.format(J))
        return train_feature, a, z, b, y_pred, J

    def NNBackward(self, x, y, a, z, b, y_pred, J):
        grad_J = 1
        # grad_y_pred = self.crossEntropyBackward(y, y_pred, J, grad_J)
        # grad_b = self.softmaxBackward(b, y_pred, grad_y_pred)
        grad_b = self.softmaxCrossentropyBackward(y, y_pred, grad_J)
        # print('d(loss)/d(softmax inputs): {}'.format(grad_b))
        grad_beta, grad_z = self.linearBackward(z, self.beta, b, grad_b)
        # print('d(loss)/d(second layer weights): {}'.format(grad_beta))
        # print('d(loss)/d(hidden layer outputs): {}'.format(grad_z))
        grad_a = self.sigmoidBackward(a, z, grad_z)
        # print('d(loss)/d(hidden layer inputs): {}'.format(grad_a))
        grad_alpha, grad_x = self.linearBackward(x, self.alpha, a, grad_a)
        # print('d(loss)/d(first layer weights): {}'.format(grad_alpha))
        return grad_alpha, grad_beta

    def SGD(self, train_feature, train_label, alpha, beta):
        x, a, z, b, y_pred, J = self.NNForward(train_feature, train_label, alpha, beta)
        grad_alpha, grad_beta = self.NNBackward(x, train_label, a, z, b, y_pred, J)
        alpha = alpha - self.lr * grad_alpha
        beta = beta - self.lr * grad_beta
        return alpha, beta

    def train(self):
        crossEntropy = []
        train_mce = 0
        test_mce = 0
        for e in range(self.epochs):
            print("Running epoch {}".format(e))
            for i in range(len(train_feature)):
                self.alpha, self.beta = self.SGD(self.train_feature[i], self.train_label[i], self.alpha, self.beta)
                # print('First Layer Weights:\n {}\n'.format(self.alpha))
                # print('Second Layer Weights:\n {}\n'.format(self.beta))
            # Evaluate training mean cross-entropy JD(alpha; beta)
            train_mce = self.meanCrossEntropy(self.train_feature, self.train_label, self.alpha, self.beta)
            # print("Epoch= {}, crossentropy(train): {}".format(e, train_mce))
            crossEntropy.append("Epoch= {}, crossentropy(train): {}".format(e, train_mce))
            self.train_mean_ce.append(train_mce)
            # # Evaluate test mean cross-entropy JDt(alpha; beta)
            test_mce = self.meanCrossEntropy(self.test_feature, self.test_label, self.alpha, self.beta)
            crossEntropy.append("Epoch= {}, crossentropy(test): {}".format(e, test_mce))
            self.test_mean_ce.append(test_mce)
            # print("Epoch= {}, crossentropy(test): {}".format(e, test_mce))
        return self.alpha, self.beta, crossEntropy, train_mce, test_mce, self.train_mean_ce, self.test_mean_ce

    def meanCrossEntropy(self, feature, label, alpha, beta):
        mean_cross_entropy = 0
        for i in range(len(feature)):
            _, _, _, _, _, J = self.NNForward(feature[i], label[i], alpha, beta)
            mean_cross_entropy += J
        return mean_cross_entropy / len(feature)

    def init_param(self, init_flag):
        if init_flag == 1:
            alpha = np.random.uniform(-0.1, 0.1, ((hidden_units, 1 + len(self.train_feature[0]))))
            for i in alpha:
                i[0] = 0
            beta = np.random.uniform(-0.1, 0.1, (10, hidden_units + 1))
            for j in beta:
                j[0] = 0

        else:
            alpha = np.zeros((hidden_units, 1 + len(self.train_feature[0])))
            beta = np.zeros((10, hidden_units + 1))

        return alpha, beta

    def predict(self, x):
        result = []
        for i in x:
            a = self.linearForward(i, self.alpha)
            z = self.sigmoidForward(a)
            b = self.linearForward(z, beta)
            y_pred = self.softmaxForward(b)
            result.append(np.argmax(y_pred))
        return result

    def errorRate(self, y_pred, y):
        error = 0
        y = [np.argmax(i) for i in y]
        for i in range(len(y_pred)):
            if y_pred[i] != y[i]:
                error += 1
        return error / len(y_pred)


# load input
if __name__ == "__main__":
    train_file = 'largeTrain.csv'
    test_file = 'largeTest.csv'
    # train_out = sys.argv[3]
    # test_out = sys.argv[4]
    # metrics_out = sys.argv[5]
    num_epochs = 100
    hidden_units = 50
    init_flag = 1
    learning_rate = 0.01

    # load date into array
    arr_train = np.genfromtxt(train_file, dtype=int, delimiter=',', skip_header=0)
    arr_test = np.genfromtxt(test_file, dtype=int, delimiter=',', skip_header=0)

    train_feature = np.asarray([i[1:] for i in arr_train])
    train_label = np.zeros((len(train_feature), 10))
    test_feature = np.asarray([i[1:] for i in arr_test])
    test_label = np.zeros((len(test_feature), 10))

    # one-hot encoding of labels
    for i in range(len(arr_train)):
        train_label[i][arr_train[i][0]] = 1

    for i in range(len(arr_test)):
        test_label[i][arr_test[i][0]] = 1


    # # code for drawing graphs for question 1
    # train_mce = []
    # test_mce=[]
    # for hidden_units in [5, 20, 50, 100, 200]:
    #     print("Running hidden units: ", hidden_units)
    #     neural_net = neuralNetwork(train_feature, train_label, test_feature, test_label, init_flag, learning_rate,
    #                                hidden_units, num_epochs)
    #     alpha, beta, crossEntropy, train_Mean_CrossEntropy, test_Mean_CrossEntropy, _, _ = neural_net.train()
    #     # print('\n\nFinal alpha: {}\n\n'.format(alpha))
    #     # print('Final beta: {}'.format(beta))
    #     # train_pred = neural_net.predict(train_feature)
    #     # train_error = neural_net.errorRate(train_pred, train_label)
    #     # print("\nTrain error: {}".format(train_error))
    #     # test_pred = neural_net.predict(test_feature)
    #     # test_error = neural_net.errorRate(test_pred, test_label)
    #     # print("Test error: {}".format(test_error))
    #     print(train_Mean_CrossEntropy)
    #     print(test_Mean_CrossEntropy)
    #     train_mce.append(train_Mean_CrossEntropy)
    #     test_mce.append(test_Mean_CrossEntropy)
    #
    # plt.plot([5, 20, 50, 100, 200],train_mce, label="Training Mean Cross Entropy")
    # plt.plot([5, 20, 50, 100, 200],test_mce, label="Test Mean Cross Entropy")
    # plt.legend()
    # plt.ylabel('Average Cross Entropy')
    # plt.xlabel("Number of Hidden Units")
    # plt.title("Average Cross Entropy of training and test set vs Num of Hidden Units")
    # plt.show()
    #
    # np.savetxt("train_mce.txt", train_mce, delimiter=",", fmt="%s")
    # np.savetxt("test_mce.txt", test_mce, delimiter=",", fmt="%s")
    #

    # code for second part graphs
    hidden_units = 50
    for learning_rate in [0.1, 0.01, 0.001]:
        neural_net = neuralNetwork(train_feature, train_label, test_feature, test_label, init_flag, learning_rate,
                                   hidden_units, num_epochs)
        alpha, beta, crossEntropy, _, _, train_Mean_CrossEntropy, test_Mean_CrossEntropy = neural_net.train()
        print('\n\nFinal alpha: {}\n\n'.format(alpha))
        print('Final beta: {}'.format(beta))
        # train_pred = neural_net.predict(train_feature)
        # train_error = neural_net.errorRate(train_pred, train_label)
        # print("\nTrain error: {}".format(train_error))
        # test_pred = neural_net.predict(test_feature)
        # test_error = neural_net.errorRate(test_pred, test_label)
        # print("Test error: {}".format(test_error))
        print(train_Mean_CrossEntropy)
        print(test_Mean_CrossEntropy)
        plt.plot([i for i in range(1,101)], train_Mean_CrossEntropy, label="Training Mean Cross Entropy")
        plt.plot([i for i in range(1,101)], test_Mean_CrossEntropy, label="Test Mean Cross Entropy")
        plt.legend()
        plt.ylabel('Avg Cross Entropy')
        plt.xlabel("# of Epochs")
        plt.title("AvgCross Entropy VS # of Epochs in learning rate "+str(learning_rate))
        plt.show()


    #0.01
    #[0.9368930657979807, 0.6845663623803628, 0.5966663776519165, 0.543078075510235, 0.5043017670118647, 0.4735240663568519, 0.4476563545152179, 0.42511730690519356, 0.40501334957585766, 0.38679232490247833, 0.3700803366101059, 0.35460915018372763, 0.3401844624316139, 0.3266677203549405, 0.3139609565344498, 0.301992576960435, 0.29070572167614156, 0.2800510092186705, 0.26998330189410347, 0.26046073920139373, 0.2514445522676824, 0.24289895651041915, 0.2347910634892669, 0.22709100139814628, 0.21977234093558162, 0.21281259995528717, 0.20619315840651298, 0.1998979571680936, 0.19391143925012808, 0.18821720585655122, 0.18279808657727026, 0.17763692611913845, 0.17271726705407156, 0.1680236983585156, 0.16354196882472818, 0.15925898924281234, 0.15516278712211903, 0.1512424329409855, 0.14748794067098178, 0.14389014548375134, 0.14044056621201284, 0.13713126504069914, 0.13395472200566977, 0.1309037439163894, 0.1279714203445033, 0.12515112409291834, 0.12243654001658627, 0.11982170243480053, 0.11730102618035648, 0.11486932340976388, 0.11252180409683761, 0.11025406229736652, 0.10806205297952383, 0.10594206502260572, 0.10389069460437184, 0.10190482043842364, 0.09998157986891197, 0.09811834396477738, 0.09631269047040233, 0.09456237478807825, 0.09286530015407295, 0.09121948857921781, 0.08962305428215857, 0.08807418155504498, 0.08657110910148451, 0.08511212235400135, 0.0836955538465946, 0.0823197898552711, 0.080983280189807, 0.07968454789804383, 0.07842219659797112, 0.07719491447976458, 0.07600147506266057, 0.07484073530270294, 0.07371163172927142, 0.07261317515973956, 0.07154444437828592, 0.07050457905818999, 0.06949277217985207, 0.0685082622283338, 0.06755032550182341, 0.06661826888310603, 0.06571142339453179, 0.06482913877246509, 0.0639707791801825, 0.06313572005740573, 0.06232334600530353, 0.061533049541611984, 0.06076423053309499, 0.06001629611492273, 0.05928866092845672, 0.05858074754066061, 0.057891986942764824, 0.05722181905830026, 0.056569693218644676, 0.05593506858646588, 0.05531741452338428, 0.054716210907989005, 0.05413094841465299, 0.05356112876366352]
    #[0.9784579317240814, 0.7486889354810301, 0.6729792645739653, 0.6287038731813644, 0.597722094149987, 0.5738343623642645, 0.554234829733252, 0.5375135237527625, 0.5228754209402496, 0.5098257344071264, 0.4980310727847957, 0.4872582668290913, 0.47734715965779817, 0.4681939752791301, 0.45973561763214116, 0.4519320232340662, 0.4447508124667333, 0.4381592148669848, 0.43212200267678164, 0.42660211975635376, 0.42156245165607453, 0.4169678607712022, 0.41278661275468465, 0.4089907809339271, 0.40555592865999057, 0.4024606020525897, 0.3996857106835287, 0.39721353898949857, 0.3950266688129256, 0.39310765197643455, 0.39143959993142013, 0.39000696808315644, 0.3887959448409715, 0.3877943901700569, 0.38699151465520254, 0.386377482755419, 0.38594304933223994, 0.38567927983192196, 0.38557737161690875, 0.38562857181562493, 0.3858241646822141, 0.38615548448711867, 0.38661391374568116, 0.3871908559801275, 0.38787770903231933, 0.38866587966353383, 0.3895468601222683, 0.3905123513689057, 0.39155439624096167, 0.39266549014583574, 0.39383865452560424, 0.3950674733047631, 0.3963460987924981, 0.3976692333401655, 0.3990320907207415, 0.40043033980014114, 0.40186003388672115, 0.4033175316342976, 0.4047994178307895, 0.40630243324863036, 0.4078234214688391, 0.40935929776081476, 0.41090704161626485, 0.41246371113458574, 0.4140264746010602, 0.41559265265173223, 0.41715976371771835, 0.41872556609732403, 0.4202880916502303, 0.4218456680194161, 0.4233969278775918, 0.4249408048740039, 0.4264765169648283, 0.4280035388065923, 0.429521565789919, 0.43103047290242413, 0.43253027182930526, 0.4340210695394146, 0.4355030311274634, 0.4369763489620518, 0.4384412193088252, 0.4398978266686165, 0.4413463352207233, 0.44278688610890066, 0.4442195989305882, 0.44564457571, 0.4470619058175495, 0.44847167066234284, 0.44987394742976305, 0.45126881156576565, 0.452656338051335, 0.45403660172897525, 0.45540967703667434, 0.4567756374972395, 0.4581345552375585, 0.45948650070936475, 0.4608315426794178, 0.46216974847230397, 0.46350118439233, 0.4648259162242935]





    # # write test and train prediction label output
    # np.savetxt(train_out, train_pred, delimiter=",", fmt="%s")
    # np.savetxt(test_out, test_pred, delimiter=",", fmt="%s")
    #
    # # write metrics output file
    # with open(metrics_out, 'w') as file:
    #     for i in crossEntropy:
    #         file.write(i + "\n")
    #     file.write('error(train): ' + str(train_error) + '\n')
    #     file.write('error(test): ' + str(test_error))



