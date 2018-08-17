from scipy.io import loadmat
from sklearn.linear_model import SGDClassifier  
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import numpy.random as rnd

mnist_raw = loadmat("datasets/mnist-original.mat")
mnist = {
    "data": mnist_raw["data"].T,
    "target": mnist_raw["label"][0],
    "COL_NAMES": ["label", "data"],
    "DESCR" : "mldata.org dataset: mnist-orignal",
}

#print(mnist)

X, Y = mnist["data"], mnist["target"]
# print(X.shape)
# print(Y.shape)

some_digit = X[36000]
# print(some_digit)
some_digit_image = some_digit.reshape(28,28)

plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation="nearest")

plt.axis("off")
# plt.show()
# print(Y[36000])


X_train, Y_train, X_test, Y_test = X[:60000], Y[:60000], X[60000:], Y[60000:]

shuffle_index = np.random.permutation(60000)
print(shuffle_index)
X_train, Y_train = X_train[shuffle_index], Y[shuffle_index]


y_train_5 = (Y_train == 5)
y_test_5 = (Y_test == 5)



# print(y_train_5.shape, y_train_5)
# print(y_test_5.shape, y_test_5)

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

# print(sgd_clf.predict([some_digit]))

# print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))




skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    
    X_train_folds = X_train[train_index]
    y_train_folds = (y_train_5[train_index])
    X_test_fold = X_train[test_index]
    y_test_folds  = (y_train_5[test_index])

    clone_clf.fit(X_train_folds, y_train_folds)

    y_pred = clone_clf.predict(X_test_fold)

    n_correct = sum(y_test_folds == y_pred)

    print(n_correct / float(len(y_pred)))	

class Never5Classifier(BaseEstimator):

	def fit(self, X, y=None):
		pass

	def predict(self, X):
		return np.zeros((len(X), 1), dtype=bool)

# never_5_clf = Never5Classifier()

# print(never_5_clf.predict([0,0,1,1,0]))


# print(cross_val_score(
# 	never_5_clf,
# 	X_train,
# 	y_train_5,
# 	cv=3,
# 	scoring="accuracy"))

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

# print(y_train_pred)
# print(y_train_5)

# print(confusion_matrix(y_train_5, y_train_pred))
# print("precision:\n",precision_score(y_train_5, y_train_pred))
# print("recall:\n",recall_score(y_train_5, y_train_pred))

# print("f1:\n", f1_score(y_train_5, y_train_pred))


y_scores = sgd_clf.decision_function([some_digit])
# print(y_scores)

threshold = 0
y_some_digit_pred = (y_scores > threshold)
# print(y_some_digit_pred)

threshold =  200000
y_some_digit_pred = (y_scores > threshold)
# print(y_some_digit_pred)


y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1],"b--",label="Precision")
    plt.plot(thresholds, recalls[:-1],"g-",label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])

# plt.figure(figsize=(10, 4))
# plot_precision_vs_recall(precisions, recalls)
# plt.show()


y_train_pred_90 = (y_scores > 50000)
# print(y_train_pred_90.shape, y_train_pred_90)

# print("precision:\n",precision_score(y_train_5, y_train_pred_90))
# print("recall:\n", recall_score(y_train_5, y_train_pred_90))
# print("f1:\n", f1_score(y_train_5, y_train_pred))

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr,tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

plot_roc_curve(fpr, tpr)
plt.show()      

print(roc_auc_score(y_train_5, y_scores))


forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")

# y_scores_forest = y_probas_forest[:, 1]
# fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

# plt.plot(fpr,tpr, "b:", label="SGD")
# plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
# plt.legend(loc="lower right")
# plt.show() 

# sgd_clf.fit(X_train, Y_train)
# print(sgd_clf.predict([some_digit]))

# some_digit_scores = sgd_clf.decision_function([some_digit])
# print(some_digit_scores)
# print(sgd_clf.classes_)

# ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
# ovo_clf.fit(X_train, Y_train)
# print("prediciton:\n",ovo_clf.predict([some_digit]))

# forest_clf.fit(X_train, Y_train)
# print("rediciton via Random Forest:\n",forest_clf.predict([some_digit]))
# print("probanility via Random Forest:\n",forest_clf.predict_proba([some_digit]))

print("CV score:\n",cross_val_score(sgd_clf, X_train, Y_train, cv=3, scoring="accuracy"))

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

print("CV score, scaled inputs:\n", cross_val_score(sgd_clf, X_train_scaled, Y_train, cv=3, scoring="accuracy"))


y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, Y_train, cv=3)

conf_mx = confusion_matrix(Y_train, y_train_pred)
print("confusion materix:\n", conf_mx)

# plt.matshow(conf_mx, cmap=plt.cm.gray)
# plt.show()

row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums

# np.fill_diagonal(norm_conf_mx, 0)
# plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
# plt.show()

def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size, size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")


# cl_a, cl_b  =3, 5

# X_aa = X_train[(Y_train == cl_a) & (y_train_pred == cl_a)]
# X_ab = X_train[(Y_train == cl_a) & (y_train_pred == cl_b)]
# X_ba = X_train[(Y_train == cl_b) & (y_train_pred == cl_a)]
# X_bb = X_train[(Y_train == cl_b) & (y_train_pred == cl_b)]

# plt.figure(figsize=(8,8))
# plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
# plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
# plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
# plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
# plt.show()

y_train_large =  (Y_train >= 7)
y_train_odd = (Y_train % 2 == 1)

print("large nums?\n", y_train_large)
print("odd nums?\n", y_train_odd)

y_multilabel = np.c_[y_train_large, y_train_odd]

print("combined (multilabel) ?\n", y_multilabel)

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)

print("KNN prediction of some digit: (>=7? odd?)\n", knn_clf.predict([some_digit]))


y_train_knn_pred = cross_val_predict(knn_clf, X_train, Y_train, cv=3)

print(f1_score(Y_train, y_train_knn_pred, average="macro"))

noise = rnd.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = rnd.randint(0, 100, (len(X_train), 784))
X_test_mod = X_test + noise

y_train_mod = X_train
y_test_mod = X_test

some_index = 5500

def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap= matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")

knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[some_index]])

plot_digit(clean_digit)
