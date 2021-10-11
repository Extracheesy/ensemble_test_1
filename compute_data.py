import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_breast_cancer

from sklearn.svm import SVC

def split_df_data(df):
    df_tic = pd.DataFrame()
    df_tic['tic'] = df['tic'].copy()

    tic_list = df_tic['tic'].unique()

    for tic in tic_list:
        df_ensemble = pd.DataFrame()
        df_tic = pd.DataFrame()
        df_select = df[df['tic'] == tic].copy()
        model_list = df_select['model'].unique()
        for model in model_list:
            df_model = df_select[df_select['model'] == model].copy()
            df_ensemble[model] = df_model['y_test_prob'].copy()
        df_ensemble['y_test_target'] = df_model['y_test'].copy()
        df_ensemble.to_csv(tic + '_ensemble.csv')

        X_train, y_train, X_test, y_test = get_train_test_split(df_ensemble, 21)

        classification_RT(X_train, y_train, X_test, y_test, tic)
        #classification_SVC(X_train, y_train, X_test, y_test, tic)



def get_train_test_split(df, test_size):
    split_index = len(df) - test_size
    df_train = df.iloc[:split_index, :]
    df_test = df.iloc[split_index+1:, :]
    """
    X_train, X_test, y_train, y_test = train_test_split(df,
                                                        np.ravel(df_target),
                                                        test_size=0.20,
                                                        random_state=101)
    """
    y_train = df_train['y_test_target'].copy()
    columns = ['y_test_target']
    X_train = df_train.drop(columns, axis=1).copy()

    y_test = df_test['y_test_target'].copy()
    columns = ['y_test_target']
    X_test = df_test.drop(columns, axis=1).copy()

    return X_train, y_train, X_test, y_test

def classification_SVC(X_train, y_train, X_test, y_test, tic):
    # train the model on train set
    model = SVC()
    model.fit(X_train, y_train)

    # print prediction results
    predictions = model.predict(X_test)
    print("classification NO GridSearchCV: ",tic)
    print(classification_report(y_test, predictions))

    # defining parameter range
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']}

    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1)

    # fitting the model for grid search
    grid.fit(X_train, y_train)

    grid.fit(X_train, y_train)

    # print best parameter after tuning
    print(grid.best_params_)

    # print how our model looks after hyper-parameter tuning
    print(grid.best_estimator_)

    grid_predictions = grid.predict(X_test)

    print("classification WITH GridSearchCV: ", tic)
    # print classification report
    print(classification_report(y_test, grid_predictions))


def classification_RT(X_train, y_train, X_test, y_test, tic):

    #cv = MultipleTimeSeriesCV(n_splits=10, train_period_length=60,
    #                          test_period_length=6, lookahead=1)

    model = RandomForestClassifier(random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)
    # print prediction results
    predictions = model.predict(X_test)
    print("classification NO GridSearchCV: ",tic)
    print(classification_report(y_test, predictions))

    param_grid = {'n_estimators': [50, 100, 200, 500],
                  'max_features': ['auto', 'sqrt', 'log2'],
                  'max_depth': [5, 10, 15, None],
                  'criterion': ['gini', 'entropy'],
                  'min_samples_leaf': [5, 25, 100]}

    grid = GridSearchCV(estimator=model,
                        param_grid=param_grid,
                        scoring='roc_auc',
                        #scoring='accuracy',
                        #n_jobs=-1,
                        #cv=5,
                        refit=True,
                        return_train_score=True,
                        verbose=1)

    # fitting the model for grid search
    grid.fit(X_train, y_train)

    # print best parameter after tuning
    print(grid.best_params_)

    # print how our model looks after hyper-parameter tuning
    print(grid.best_estimator_)

    grid_predictions = grid.predict(X_test)

    print("classification WITH GridSearchCV: ", tic)
    # print classification report
    print(classification_report(y_test, grid_predictions))
