import pandas as pd
import numpy as np
from sklearn.cross_validation import  train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn import tree
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.grid_search import  GridSearchCV

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

train_df.columns = ['family', 'product', 'steel', 'carbon', 'hardness', 'temper_rolling', 'condition', 'formability',
                   'strength', 'non-ageing', 'surface-finish', 'surface-quality', 'enamelability', 'bc', 'bf', 'bt',
                   'bw/me', 'bl', 'm', 'chrom', 'phos', 'cbond', 'marvi', 'exptl', 'ferro', 'corr', 'blue/bright/varn/clean',
                   'lustre', 'jurofm', 's', 'p', 'shape', 'thick', 'width', 'len', 'oil', 'bore', 'packing', 'classes']


train_df.replace(to_replace='?', value=np.nan, inplace=True)
train_df.dropna(thresh=730, axis=1, inplace=True)
train_df.dropna(thresh=2, axis=0, inplace=True)
y = train_df.iloc[:, -1].values

X = pd.get_dummies(train_df.iloc[:, :-2])

# print(X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

clf = tree.DecisionTreeClassifier()
# clf.fit(X_train, y_train)

# y_pred = clf.predict(X_test)
# print(len(y_pred), len(y_test))

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

gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)

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



# continuous_df = train_df.loc[:, ['carbon', 'hardness', 'strength', 'thick', 'width', 'len']]
# train_df.drop(columns=['carbon', 'hardness', 'strength', 'thick', 'width', 'len'], inplace=True)

# print(train_df.head())


# imr = Imputer(missing_values='NaN', strategy='most_frequent', axis=1)
# imputer_data = imr.fit(train_df)
# imputed = imr.transform(train_df.values)




