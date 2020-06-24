import pandas as pd
from texttable import Texttable
from sklearn.model_selection import train_test_split
from DataCleaning import DataClean
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

def reform_data(X_train,Test_set):

    # ///// 1. Get Titles as number ////////////////
    titles_num, title_dict = DataClean.Names_to_num_with_dict(X_train)
    X_train['Titels'] = titles_num

    titles_num = DataClean.Names_to_num(Test_set, title_dict)
    Test_set['Titels'] = titles_num

    # ///// 2. Age Edition ////////////////
    # Missing age values
    mean_age_dict = round(X_train.groupby('Titels')['Age'].mean(), 0).to_dict()
    for age in mean_age_dict:
        X_train.loc[(X_train['Age'].isnull()) & (X_train['Titels'] == age), 'Age'] = mean_age_dict[age]
        Test_set.loc[(Test_set['Age'].isnull()) & (Test_set['Titels'] == age), 'Age'] = mean_age_dict[age]
    # View the distribution
    # chart = sns.catplot(x="Age", kind="count", data=X_train, height=4, aspect=1)
    # chart.set_xticklabels(rotation=90)
    # plt.show()

    # bring into bins
    # We split the feature Age in five categories, where category 1 refers to children (age<15)
    X_train["Age"] = pd.cut(X_train["Age"], 85 // 5, labels=range(1, (85 // 5) + 1))
    Test_set["Age"] = pd.cut(Test_set["Age"], 85 // 5, labels=range(1, (85 // 5) + 1))

    # View the distribution
    # chart = sns.catplot(x="Age", kind="count", data=X_train, height=4, aspect=1)
    # chart.set_xticklabels(rotation=90)
    # plt.show()
    # ///// 3. Add new feature: Family size ////////////////
    X_train["Family Size"] = X_train["SibSp"] + X_train["Parch"] + 1
    X_train["Fare"] = X_train["Fare"] / X_train["Family Size"]
    X_train["Fare"] = pd.cut(X_train["Fare"], 550 // 5, labels=range(1, (550 // 5) + 1))

    Test_set["Family Size"] = Test_set["SibSp"] + Test_set["Parch"] + 1
    Test_set["Fare"] = Test_set["Fare"] / Test_set["Family Size"]
    Test_set["Fare"] = pd.cut(Test_set["Fare"], 550 // 5, labels=range(1, (550 // 5) + 1))

    # ///// 4. Edit Cabins and Add new feature: Cabin numbers ////////////////
    Cabin_num, Cabin_dict, numer_of_cabins = DataClean.Cabin_to_num_with_dict(X_train)
    X_train['Cabin'] = Cabin_num
    X_train['#Cabins'] = numer_of_cabins

    Cabin_num, numer_of_cabins = DataClean.Cabin_to_num(Test_set, Cabin_dict)
    Test_set['Cabin'] = Cabin_num
    Test_set['#Cabins'] = numer_of_cabins

    # ///// 4. Edit Sex column ////////////////
    sex_dict = {'male': 1, 'female': 2}
    sex_num = DataClean.Sex_to_num(X_train, sex_dict)
    X_train['Sex'] = sex_num

    sex_num = DataClean.Sex_to_num(Test_set, sex_dict)
    Test_set['Sex'] = sex_num

    # ///// 5. Edit Embarked column ////////////////
    embarked_num, embarked_dict = DataClean.Embarked_to_num_with_dict(X_train)
    X_train['Embarked'] = embarked_num

    embarked_num = DataClean.Embarked_to_num(Test_set, embarked_dict)
    Test_set['Embarked'] = embarked_num

    # ///// 6. Drop unusfull columns and check for nan values ////////////////
    X_train = X_train.drop(['Name', 'Ticket'], axis=1)
    Test_set = Test_set.drop(['Name', 'Ticket'], axis=1)
    # check missing values
    a = X_train.isnull().sum()
    if a.sum() > 0:
        print("there is still nun value in your Train Data")
    a = Test_set.isnull().sum()
    if a.sum() > 0:
        print("there is still nun value in your Test Set")

    return X_train,Test_set

class DoLogreg:
    def classify_LogReg(X_train, y_train,X_test,y_test):

        ''' logistic regresiion '''
        from sklearn.linear_model import LogisticRegression
        lgreg = LogisticRegression(max_iter=500)  # Define the model
        lgreg.fit(X_train, y_train)  # fit using the train data
        y_pred = lgreg.predict(X_test)  # predict for test data
        print("logestic Regression accuracy {}:".format(metrics.accuracy_score(y_test, y_pred)))  # •Proportion of correct predictions

    def Classify_LogReg_kfold(X, y):
            ''' KNN classification '''
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score

            lgreg = LogisticRegression(max_iter=500)  # Define the model
            scores = cross_val_score(lgreg, X, y, cv=10, scoring='accuracy')

            # use average accuracy as an estimate of out-of-sample accuracy
            print("LogReg_kfold_accuracy {}:".format(scores.mean()))

class DoKNN:
    def find_K (X_train, y_train,X_test,y_test):
        ''' KNN classification '''
        from sklearn.neighbors import KNeighborsClassifier
        # Knn parameter tunning -> parameter is k (number of clusters)
        # try k=1 through k =25 and record testing accuracy
        k_range = range(1, 26)
        scores = list()
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            scores.append(metrics.accuracy_score(y_test, y_pred))
        print(*scores)

        # plot the relationhip between K and testing accuracy
        plt.plot(k_range, scores)
        plt.xlabel('value of K for KNN')
        plt.ylabel('Testing Accuracy')
        plt.grid()
        plt.show()

    def classify_KNN(X_train, y_train,X_test,y_test,k=5):
        ''' KNN classification '''
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        print("KNN accuracy {}:".format(metrics.accuracy_score(y_test, y_pred)))

    def Classify_KNN_kfold(X, y,k=5):
        ''' KNN classification '''
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_val_score

        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
        # use average accuracy as an estimate of out-of-sample accuracy
        print("KNN _kfold_accuracy {}:".format(scores.mean()))

    def find_K_GridseacrhCV(X, y):
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import GridSearchCV
        import pandas as pd
        # define the parameter values that should be searched
        k_range = list(range(1, 31))
        param_grid = dict(n_neighbors=k_range)
        knn = KNeighborsClassifier(n_neighbors=1)
        # instantiate the grid
        grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False)
        # fit the grid with data
        grid.fit(X, y)
        # view the results as a pandas DataFrame
        pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
        # examine the first result
        print(grid.cv_results_['params'][0])
        print(grid.cv_results_['mean_test_score'][0])
        # print the array of mean scores only
        grid_mean_scores = grid.cv_results_['mean_test_score']
        print(grid_mean_scores)
        # plot the results
        plt.plot(k_range, grid_mean_scores)
        plt.xlabel('Value of K for KNN')
        plt.ylabel('Cross-Validated Accuracy')
        # examine the best model
        print(grid.best_score_)
        print(grid.best_params_)
        print(grid.best_estimator_)

class ANN:
    def simple(X,Y):
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import Dropout

        # Define and Compile
        model = Sequential()
        model.add(Dense(64, input_dim=11, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(252, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Fit the model
        model.fit(X, Y, epochs=400, batch_size=10, verbose=2)
        # Evaluate the model
        scores = model.evaluate(X, Y)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        return model

    def cross_val(X,Y):
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import Dropout
        import tensorflow as tf
        from sklearn.model_selection import StratifiedKFold
        from sklearn.model_selection import cross_val_score

        def create_model():
            # Define and Compile
            model = Sequential()
            model.add(Dense(50, input_dim=11, activation='relu'))
            model.add(Dense(20, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(8, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model

        # create classifier for use in scikit-learn
        model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model, epochs=20, batch_size=10,
                                                               verbose=2)
        # evaluate model using 10-fold cross-validation in scikit-learn
        kfold = StratifiedKFold(n_splits=10, shuffle=True)
        results = cross_val_score(model, X, Y, cv=kfold)
        print(results)

def main():
    Train_set,Test_set = [pd.read_csv('Dataset'+'/'+i, index_col=0) for i in ['train.csv', 'test.csv']]
    X_train = Train_set.drop(['Survived'],axis = 1)
    Y_train = Train_set.Survived
    """ Table of information """
    print("\n/////////////////////// Table : Summary ///////////////////////")
    t = Texttable()
    t.add_rows([
        ['# Name', ' # Shape'],
        ['X_train', X_train.shape],
        ['Y_train', Y_train.shape],
        ['X_tset', Test_set.shape]
    ])
    print(t.draw())
    print('\n Featurenames :{}'.format(X_train.columns))
    print("//////////////////////////////////////////////////////////////\n")

    X_train,Test_set = reform_data(X_train,Test_set)


    X_train_splitted, X_test_splitted, Y_train_splitted, Y_test_splitted = train_test_split(X_train, Y_train, test_size=0.15, random_state=4)

    """ Table of information """
    print("\n/////////////////////// Table : Summary After split ///////////////////////")
    t = Texttable()
    t.add_rows([
        ['# Name', ' # Shape'],
        ['X_train', X_train_splitted.shape],
        ['Y_train', Y_train_splitted.shape],
        ['X_val', X_test_splitted.shape],
        ['Y_val', Y_test_splitted.shape],
        ['X_tset', Test_set.shape]
    ])
    print(t.draw())
    print('\n Featurenames :{}'.format(X_train.columns))
    print("//////////////////////////////////////////////////////////////\n")

    DoLogreg.classify_LogReg(X_train_splitted, Y_train_splitted, X_test_splitted,Y_test_splitted)
    DoLogreg.Classify_LogReg_kfold(X_train, Y_train)
    #DoKNN.find_K(X_train, Y_train,X_test,Y_test)
    DoKNN.classify_KNN(X_train_splitted, Y_train_splitted,X_test_splitted,Y_test_splitted, k=13)
    DoKNN.Classify_KNN_kfold(X_train, Y_train, k=7)
    #DoKNN.find_K_GridseacrhCV(X_train, Y_train)

    #model = ANN.simple(X_train, Y_train)
    #model.save('my_model.h5')
    from keras.models import load_model
    # load model
    model = load_model('my_model.h5')
    predicted = model.predict(Test_set)
    predicted[predicted>0.5]=1
    predicted[predicted<0.5]=0
    predicted = predicted.astype('uint8')
    submission = pd.read_csv('Dataset/gender_submission.csv')
    submission['Survived'] = predicted
    submission.to_csv('submission.csv', index=False)

    import io
    import requests
    url = "https://github.com/thisisjasonjafari/my-datascientise-handcode/raw/master/005-datavisualization/titanic.csv"
    s = requests.get(url).content
    c = pd.read_csv(io.StringIO(s.decode('utf-8')))

    test_data_with_labels = c
    print(2)

if __name__ =='__main__':
    main()