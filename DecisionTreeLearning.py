import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import tree
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
from sklearn.base import TransformerMixin

# Reading Data
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

train_df.head()
# train_df.info()

# add column names to data frames to process them easier
train_df.columns = ['family', 'product', 'steel', 'carbon', 'hardness', 'temper_rolling', 'condition', 'formability',
                   'strength', 'non-ageing', 'surface-finish', 'surface-quality', 'enamelability', 'bc', 'bf', 'bt',
                   'bw/me', 'bl', 'm', 'chrom', 'phos', 'cbond', 'marvi', 'exptl', 'ferro', 'corr', 'blue/bright/varn/clean',
                   'lustre', 'jurofm', 's', 'p', 'shape', 'thick', 'width', 'len', 'oil', 'bore', 'packing', 'classes']

test_df.columns = ['family', 'product', 'steel', 'carbon', 'hardness', 'temper_rolling', 'condition', 'formability',
                   'strength', 'non-ageing', 'surface-finish', 'surface-quality', 'enamelability', 'bc', 'bf', 'bt',
                   'bw/me', 'bl', 'm', 'chrom', 'phos', 'cbond', 'marvi', 'exptl', 'ferro', 'corr', 'blue/bright/varn/clean',
                   'lustre', 'jurofm', 's', 'p', 'shape', 'thick', 'width', 'len', 'oil', 'bore', 'packing']

# handle missing data
train_df.replace(to_replace='?', value=np.nan, inplace=True)
train_df.dropna(thresh=500, axis=1, inplace=True)
train_df.dropna(thresh=2, axis=0, inplace=True)
train_df = train_df[train_df.classes != 'U']

test_df.replace(to_replace='?', value=np.nan, inplace=True)

# print(train_df.info())


class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.
        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


train_df = DataFrameImputer().fit_transform(train_df)
test_df = DataFrameImputer().fit_transform(test_df)


# target values
y = train_df.iloc[:, -1].values

# train and test
X = train_df.iloc[:, :-1]
X_final_test = test_df.loc[:, X.columns]
# print(X.columns)
# print(X_final_test.columns)
# print(train_df.columns)

# one-hot encoding
X = pd.get_dummies(X)
X_final_test = pd.get_dummies(X_final_test)
# print(X.columns)
# print(X_final_test.columns)


def plot_correlation_map( df ):
    corr = train_df.corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr,
        cmap = cmap,
        square=True,
        cbar_kws={ 'shrink' : .9 },
        ax=ax,
        annot = True,
        annot_kws = { 'fontsize' : 12 }
    )
    plt.show()


# plot_correlation_map(train_df)


# split X to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

clf = tree.DecisionTreeClassifier()

#clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print(np.count_nonzero(y == '3'))
# print(accuracy_score(y_test, y_pred))


# scores = cross_val_score(estimator=clf,
#     X=X,
#     y=y,
#     cv=10,
#     n_jobs=1)
# print('CV accuracy scores: %s' % scores)
# print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),
# np.std(scores)))

# hyper parameters tuning
param_grid = [{'criterion': ['gini'],
               'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
               'min_samples_split': [2, 3, 4, 5]},
              {'criterion': ['entropy'],
               'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
               'min_samples_split': [2, 3, 4, 5]}
              ]

gs = GridSearchCV(estimator=clf,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10,
                  n_jobs=-1)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)


clf = gs.best_estimator_
clf.fit(X, y)

y_pred = clf.predict(X_final_test)

# there are many classes with label 3. This is a weird data sets. I think classifier is not general for all data sets
# and maybe distribution of data is not good. we need more data!!!
print(np.count_nonzero(y == '3'))
print(np.count_nonzero(y_pred == '3'))


output = pd.DataFrame(y_pred)
output.to_csv('result.csv', index=False)





