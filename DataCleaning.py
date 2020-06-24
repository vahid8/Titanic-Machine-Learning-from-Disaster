import numpy as np
from typing import Dict,List
import pandas as pd
import re

class DataClean:
    def Sex_to_num(X_train, sex_dict):
        sex_column = X_train.Sex
        all_names_num = list()
        for item in sex_column:
            all_names_num.append(sex_dict[item])

        return np.array(all_names_num, dtype=np.uint8)

    def Embarked_to_num_with_dict(X_train:pd.DataFrame)->(np.ndarray,Dict):

        # ///////////// Get all available titels and transform them into numbers
        all_names = X_train.Embarked
        Titels = dict()
        all_names_num = list()
        ident = 1
        for t in all_names:

            if t in Titels:
                all_names_num.append(Titels[t][0])
                Titels[t][1]+=1
            else:
                Titels[t] = [ident,1]
                all_names_num.append(Titels[t][0])
                ident += 1

        print("Dictionary of avilable titles {} : \n".format(len(Titels)))
        for key, value in Titels.items():
            print(key, ' : ', value)

        return np.array(all_names_num,dtype= np.uint8),Titels

    def Embarked_to_num(X_train:pd.DataFrame,Titels:Dict)->(np.ndarray,Dict):

        # ///////////// Get all available titels and transform them into numbers
        all_names = X_train.Embarked
        all_names_num = list()
        ident = 1
        for t in all_names:

            try:
                all_names_num.append(Titels[t][0])
                Titels[t][1]+=1
            except:
                print("Error in key values of Emabrket dict")

        print("Dictionary of avilable titles {} : \n".format(len(Titels)))
        for key, value in Titels.items():
            print(key, ' : ', value)

        return np.array(all_names_num,dtype= np.uint8)


    def replace_titles(title):
        if title in ['Don','Dona', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col', 'Sir']:
            return 'Mr'
        elif title in ['Countess', 'Mme', 'Lady', 'theCountess']:
            return 'Mrs'
        elif title in ['Mlle', 'Ms']:
            return 'Miss'
        elif title == 'Dr':
            return 'Mr'
        else:
            return title


    def Names_to_num_with_dict(X_train:pd.DataFrame)->(np.ndarray,Dict):

        # ///////////// Get all available titels and transform them into numbers
        all_names = X_train.Name
        Titels = dict()
        all_names_num = list()
        ident = 1
        for item in all_names:
            t = item.split(',')[1]
            t = t.split('.')[0]
            t = t.replace(' ','')
            t = DataClean.replace_titles(t)
            if t in Titels:
                all_names_num.append(Titels[t][0])
                Titels[t][1]+=1
            else:
                Titels[t] = [ident,1]
                all_names_num.append(Titels[t][0])
                ident += 1

        print("Dictionary of avilable titles {} : \n".format(len(Titels)))
        for key, value in Titels.items():
            print(key, ' : ', value)





        return np.array(all_names_num,dtype= np.uint8),Titels

    def Names_to_num(X_test,Titels:Dict):

        all_names = X_test.Name
        all_names_num = list()
        for item in all_names:
            t = item.split(',')[1]
            t = t.split('.')[0]
            t = t.replace(' ', '')
            t = DataClean.replace_titles(t)
            try :
                all_names_num.append(Titels[t][0])
            except:
                print("Error in keywords")

        return np.array(all_names_num, dtype=np.uint8)

    def Cabin_to_num_with_dict(X_train:pd.DataFrame)->(np.ndarray,Dict):

        # ///////////// Get all available titels and transform them into numbers
        all_cabins = X_train.Cabin
        Cabins = dict()
        all_cabins_num = list()
        all_number_of_cabins = list()
        ident = 1
        for t in all_cabins:

            if t is not np.nan :
                t = re.split("[^a-zA-Z]*", t)
                t = [i for i in t if len(i)>0]
                if len(t)==1:
                    num_of_cabins =len(t)
                else:
                    print(t)
                    num_of_cabins = len(t)

                t = t[0]

            else:
                num_of_cabins = 0


            if t in Cabins:
                all_cabins_num.append(Cabins[t][0])
                all_number_of_cabins.append(num_of_cabins)
                Cabins[t][1]+=1
            else:
                Cabins[t] = [ident,1]
                all_cabins_num.append(Cabins[t][0])
                all_number_of_cabins.append(num_of_cabins)
                ident += 1

        print("Dictionary of avilable titles {} : \n".format(len(Cabins)))
        for key, value in Cabins.items():
            print(key, ' : ', value)


        return np.array(all_cabins_num,dtype= np.uint8),Cabins,np.array(all_number_of_cabins,dtype= np.uint8)

    def Cabin_to_num(X_train,Cabins:Dict):

        # ///////////// Get all available titels and transform them into numbers
        all_cabins = X_train.Cabin
        all_cabins_num = list()
        all_number_of_cabins = list()
        ident = 1
        for t in all_cabins:

            if t is not np.nan:
                t = re.split("[^a-zA-Z]*", t)
                t = [i for i in t if len(i) > 0]
                if len(t) == 1:
                    num_of_cabins = len(t)
                else:
                    print(t)
                    num_of_cabins = len(t)

                t = t[0]

            else:
                num_of_cabins = 0

            try :
                all_cabins_num.append(Cabins[t][0])
                all_number_of_cabins.append(num_of_cabins)
                Cabins[t][1] += 1
            except:
                print("key value error in Cabin Dict")

        print("Dictionary of avilable titles {} : \n".format(len(Cabins)))
        for key, value in Cabins.items():
            print(key, ' : ', value)

        return np.array(all_cabins_num, dtype=np.uint8), np.array(all_number_of_cabins, dtype=np.uint8)
