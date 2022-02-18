import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import math, sys, cPickle as pickle
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

dirRaw = "/scratch3/zongmiy/zm/"
dirInputData = dirRaw + str(sys.argv[1]) + "/"

### For RNN
# file to store train chunks and test data
chkTrainFile = dirInputData+"chunkTrainData_"
testFile = dirInputData+"testData.npz"
# file to store accurs
rnnResultFile = dirInputData+'rnnResults'

nChunks = 3  # modify by hand
layer_units = int(sys.argv[1])
hidden_states = 128
batch_size = 150
n_epochs = 1
learning_rate = 0.001
one_hot_dim = 26
n_outputs = 2


def train_by_RNN(X_test, y_test, seqLentest, n_inputs=26, n_steps=333, hidden_states=128, n_outputs=2,
          batch_size=150, learning_rate=0.0001, n_epochs=5, is_training=True):
    print("X_test.shape: ", X_test.shape)
    print("y_test.shape: ", y_test.shape)
    print("seqLentest.shape: ", seqLentest.shape)

    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.int32, [None])
    seq_length = tf.placeholder(tf.int32, [None])

    def biRNN(X):
        X = tf.unstack(X, n_steps, 1)

        lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_states)
        lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_states)

        # if is_training:
        #     lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(lstm_cell_fw,input_keep_prob=dropout)
        #     lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(lstm_cell_bw,input_keep_prob=dropout)
        outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_cell_fw, lstm_cell_bw, X,
                                                                dtype=tf.float32)#, sequence_length=seq_length)

        return fully_connected(outputs[-1], n_outputs, activation_fn=None)

    logits = biRNN(X)
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

    y_ = tf.argmax(logits, 1)
    correct = tf.equal(tf.argmax(logits, 1), tf.cast(y, tf.int64))
    # correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    # prec = tf.metrics.precision(labels=y, predictions=y_)
    # recall = tf.metrics.recall(labels=y, predictions=y_)

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        init.run()
        print "start training model\n"
        for epch in range(n_epochs):
            for iChunk in range(nChunks):
                chkTFILE_i = chkTrainFile+str(iChunk)+".npz"
                npzfile = np.load(chkTFILE_i)
                X_train, y_train, seq_len_train = npzfile['XTrain'], npzfile['yTrain'], npzfile['seqLenTrain']
                trainSamples = X_train.shape[0]
                itrNum = int(math.ceil(trainSamples / batch_size))
                count = 0
                for iteration in range(itrNum):
                    if (iteration+1)*batch_size <= trainSamples:
                        X_batch, y_batch = X_train[iteration * batch_size:(iteration + 1) * batch_size], \
                                       y_train[iteration * batch_size:(iteration + 1) * batch_size]
                        seq_len_batch = seq_len_train[iteration * batch_size:(iteration + 1) * batch_size]
                    else:
                        X_batch, y_batch = X_train[iteration * batch_size:], y_train[iteration * batch_size:]
                        seq_len_batch = seq_len_train[iteration * batch_size:]

                    if (count % 100) == 0:  # output current accurs
                        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch, seq_length: seq_len_batch})
                        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test, seq_length: seqLentest})
                        print("epoch:", epch, "ith Chunk", iChunk, "iter:", count,
                              "acc train:", acc_train, " acc test:", acc_test)
                    count = count + 1

                    X_batch = X_batch.reshape((-1, n_steps, n_inputs))
                    sess.run(training_op, feed_dict={X: X_batch, y: y_batch, seq_length: seq_len_batch})
        avg_accur_train = 0
        for iChunk in range(nChunks):
            chkTFILE_i = chkTrainFile + str(iChunk) + ".npz"
            npzfile = np.load(chkTFILE_i)
            X_train, y_train, seq_len_train = npzfile['XTrain'], npzfile['yTrain'], npzfile['seqLenTrain']
            acc_train = accuracy.eval(feed_dict={X: X_train, y: y_train, seq_length: seq_len_train})
            avg_accur_train = avg_accur_train + acc_train
        avg_accur_train /= 3
        # acc_test, prec_test, recall_test = sess.run([accuracy, prec, recall],
        #                                             feed_dict={X: X_test, y: y_test, seq_length: seqLentest})
        acc_test, y, y_ = sess.run([accuracy, y, y_], feed_dict={X: X_test, y: y_test, seq_length: seqLentest})
        prec_test = metrics.precision_score(y_true=y, y_pred=y_)
        recall_test = metrics.recall_score(y_true=y, y_pred=y_)
        f1 = metrics.f1_score(y_true=y, y_pred=y_)
        fpr, tpr, thresholds = metrics.roc_curve(y_true=y, y_score=y_)

        print "final train accuracy: ", avg_accur_train
        print "final test accuracy: ", acc_test
        print "test set precision: ", prec_test
        print "test set recall: ", recall_test
        print "test set F1: ", f1
        accDic = {"avg_accur_train": avg_accur_train, "acc_test": acc_test, "pre_test": prec_test,
                  "recall_test": recall_test, "F1_test": f1, "fpr_test": fpr, "tpr_test": tpr, "thresh": thresholds}
        pickle.dump(accDic, open(rnnResultFile, "w"))

def train_by_SVM(XTrain, yTrain, Xtest, ytest):
    svc = svm.SVC()
    svc.fit(XTrain, yTrain)
    y_pred = svc.predict(Xtest)
    svm_acc = metrics.accuracy_score(y_true=ytest, y_pred=y_pred)
    svm_prec = metrics.precision_score(y_true=ytest, y_pred=y_pred)
    svm_recall = metrics.recall_score(y_true=ytest, y_pred=y_pred)
    svm_f1 = metrics.f1_score(y_true=ytest, y_pred=y_pred)
    svmDict = {"acc_test": svm_acc, "pre_test": svm_prec, "recall_test": svm_recall, "F1_test": svm_f1}
    pickle.dump(svmDict, open(SVMResultFile, "w"))
    print "final test accuracy: ", svm_acc
    print "test set precision: ", svm_prec
    print "test set recall: ", svm_recall
    print "test set F1: ", svm_f1

def train_by_LogisticRegression(XTrain, yTrain, Xtest, ytest):
    # initialize LogisticRegression Model
    lr = LogisticRegression()
    # fit data and train
    lr.fit(XTrain, yTrain)
    # predict test set
    y_pred = lr.predict(Xtest)
    lr_acc = metrics.accuracy_score(y_true=ytest, y_pred=y_pred)
    lr_prec = metrics.precision_score(y_true=ytest, y_pred=y_pred)
    lr_recall = metrics.recall_score(y_true=ytest, y_pred=y_pred)
    lr_f1 = metrics.f1_score(y_true=ytest, y_pred=y_pred)
    lrDict = {"acc_test": lr_acc, "pre_test": lr_prec, "recall_test": lr_recall, "F1_test": lr_f1}
    pickle.dump(lrDict, open(LRResultFile, "w"))
    print "final test accuracy: ", lr_acc
    print "test set precision: ", lr_prec
    print "test set recall: ", lr_recall
    print "test set F1: ", lr_f1

print "\n======start RNN model=========\n"
print "starting loading test data"
# npzfile = np.load(chkTrainFile)
# chkXTrain, chkyTrain, chkSeqTrain = npzfile['chkXTrain'], npzfile['chkyTrain'], npzfile['chkSeqTrain']
npzfile = np.load(testFile)
Xtest, ytest, seqLentest = npzfile['Xtest'], npzfile['ytest'], npzfile['seqLentest']
print "test data loaded!\n"
train_by_RNN(Xtest, ytest, seqLentest, n_inputs=one_hot_dim,
      n_steps=layer_units, hidden_states=hidden_states,
      n_outputs=n_outputs, batch_size=batch_size,
      learning_rate=learning_rate, n_epochs=n_epochs)
del Xtest, ytest, seqLentest

# npzfile = np.load(numTrainDataFile)
# XTrain, yTrain = npzfile['XTrain'], npzfile['yTrain']
# npzfile = np.load(numTestDataFile)
# Xtest, ytest = npzfile['Xtest'], npzfile['ytest']
# print "\n==========start SVM model==========\n"
# train_by_SVM(XTrain, yTrain, Xtest, ytest)
# print "\n==========start LR model==========\n"
# train_by_LogisticRegression(XTrain, yTrain, Xtest, ytest)
