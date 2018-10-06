# Back-Propagation Neural Networks

import random
import pandas as pd
import numpy
import mysql.connector
import datetime

random.seed(0)


# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b - a) * random.random() + a


# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill] * J)
    return m


# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    #return math.tanh(x)
    return 1.0 / (1.0 + numpy.e ** -x)

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    #return 1.0 - y**2
    return y * (1.0 - y)


# our activation function used in output layer is ReLu
def relu(x):
    return numpy.maximum(x, 0)

# derivative of our sigmoid function, in terms of the output (i.e. y)
def drelu(y):
    return (y > 0) * 1


# read csv for training and testing
def readCsv(fileName):
    df = pd.read_csv('C:/Users/takeshi/Documents/Tesis2/' + fileName + '.csv')

    X = df.iloc[:, 1:-1]
    x_values = X.values

    y = df.iloc[:, 9]
    y_values = y.values

    arr = []
    for x in range(5):
        arr.append([x_values[x], [y_values[x]]])

    return arr


def insert_database(query, args):
    try:
        mydb = mysql.connector.connect(
            host="localhost",
            user="youruser",
            passwd="yourpassword",
            db='yourdatabase',
            charset='utf8mb4')

        mydb.set_converter_class(NumpyMySQLConverter)
        cursor = mydb.cursor()
        cursor.execute(query, args)

        if cursor.lastrowid:
            print('last insert id', cursor.lastrowid)
        else:
            print('last insert id not found')

        mydb.commit()
    except mysql.connector.Error as error:
        print(error)

    return cursor.lastrowid

def get_training():
    ans = None
    try:
        mydb = mysql.connector.connect(
            host="localhost",
            user="youruser",
            passwd="yourpassword",
            db='yourdatabase',
            charset='utf8mb4')

        cursor = mydb.cursor()
        cursor.execute("SELECT * FROM rna_trainings WHERE is_active = 1")
        ans = cursor.fetchall()

        return ans

    except mysql.connector.Error as error:
        print(error)

    return ans

def insert_training(hidden_weights, output_weights, ni, nh, no):
    training = get_training()
    if training is not None:
        mydb = mysql.connector.connect(
            host="localhost",
            user="youruser",
            passwd="yourpassword",
            db='yourdatabase',
            charset='utf8mb4')

        cursor = mydb.cursor()
        cursor.execute("UPDATE rna_trainings SET is_active = %s WHERE id = %s", (False, training[0][0]))
        mydb.commit()

    query = "INSERT INTO rna_trainings(train_date, user_id, is_active) " \
            "VALUES(%s, %s, %s)"
    args = (datetime.datetime.now(), 1, True)
    train_id = insert_database(query, args)

    for i in range(ni):
        query = "INSERT INTO hidden_weights(neuron1, neuron2, neuron3, train_id) " \
                "VALUES(%s, %s, %s, %s)"
        args = (hidden_weights[i][nh - 3], hidden_weights[i][nh - 2], hidden_weights[i][nh - 1], train_id)
        hidden_weight_id = insert_database(query, args)

    for j in range(nh):
        query = "INSERT INTO output_weights(neuron1, train_id) " \
                "VALUES(%s, %s)"
        args = (output_weights[j][no - 1], train_id)
        output_weight_id = insert_database(query, args)


class NumpyMySQLConverter(mysql.connector.conversion.MySQLConverter):
    """ A mysql.connector Converter that handles Numpy types """

    def _float32_to_mysql(self, value):
        return float(value)

    def _float64_to_mysql(self, value):
        return float(value)

    def _int32_to_mysql(self, value):
        return int(value)

    def _int64_to_mysql(self, value):
        return int(value)


class NN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1  # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no

        # create weights
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)

        # last change in weights for momentum
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni - 1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni - 1):
            self.ai[i] = sigmoid(inputs[i])
            #self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)
            #self.ah[j] = leakrelu(sum)

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            #self.ao[k] = sigmoid(sum)
            self.ao[k] = relu(sum)

        return self.ao[:]

    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k] - self.ao[k]
            output_deltas[k] = drelu(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N * change + M * self.co[j][k]
                self.co[j][k] = change

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N * change + M * self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    def test(self, patterns):
        for p in patterns:
            result = int(round(numpy.array(self.update(p[0])) * 1000.0))
            real_effort = int(numpy.array(p[1]) * 1000.0)
            print(p[0], '->', result, 'Real Effort: ', real_effort, 'Error: ', abs((real_effort - result) * 100.0))

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=1000, N=0.5, M=0.1):
        # N: learning rate
        # M: momentum factor
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
            if i % 100 == 0:
                print('error %-.5f' % error)

        insert_training(self.wi, self.wo, self.ni, self.nh, self.no)


def demo():
    # Teach network XOR function
    pat = readCsv('UCP_Dataset_test2_1')
    # pat = readCsv('UCP_Dataset_test2')

    # create a network with two input, two hidden, and one output nodes
    n = NN(8, 3, 1)

    # train it with some patterns
    n.train(pat)
    # test it
    n.test(pat)


if __name__ == '__main__':
    demo()