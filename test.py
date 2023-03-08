import os
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"

from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Проерка версий библиотек
# Версия Python
import sys
print('Python: {}'.format(sys.version))

# Загрузка scipy
import scipy
print('scipy: {}'.format(scipy.__version__))

# Загрузка numpy
import numpy
print('numpy: {}'.format(numpy.__version__))

# Загрузка matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))

# Загрузка pandas
import pandas
print('pandas: {}'.format(pandas.__version__))

# Загрукзка scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))