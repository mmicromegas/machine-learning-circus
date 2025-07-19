# https://github.com/MAIF/shapash/blob/master/tutorial/tutorial01-Shapash-Overview-Launch-WebApp.ipynb
# https://github.com/MAIF/shapash/blob/master/tutorial/tutorial03-Shapash-overview-model-in-production.ipynb
# https://medium.com/@amitjain2110/shapash-machine-learning-interpretable-understandable-ef74012eb162
# https://www.analyticsvidhya.com/blog/2021/08/complete-guide-on-how-to-use-lightgbm-in-python/

import pandas as pd
from category_encoders import OrdinalEncoder
from lightgbm import LGBMRegressor
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from shapash.data.data_loader import data_loading
from time import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import os

import warnings
warnings.filterwarnings('ignore')

from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)

#import tensorflow as tf


def timer_func(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        # print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s') # in secs
        print(f'Function {func.__name__!r} executed in {(t2 - t1) / 60.:.2f}m')  # in mins
        return result

    return wrap_func


@timer_func
def main():
    # check python version
    if sys.version_info[0] < 3:
        print("Python " + str(sys.version_info[0]) + "  is not supported. EXITING.")
        sys.exit()

    # create os independent path and read parameter file
    paramFile = os.path.join('PARAMS', 'param.nn1')
    params = ReadParamsNN(paramFile)

    # Load Data and Review content
    data_folder = params.getForProp('prop')['data_folder']
    file_to_open = os.path.join(data_folder, params.getForProp('prop')['input_data'])
    #file_to_open = os.path.join(data_folder, "df_wiser_data_with_SentAndTime_only2023.csv")
    csat_data_raw = pd.read_csv(file_to_open)

    csat_data = csat_data_raw.dropna()

    # select list of features
    list_of_features_and_label = ["impact", "urgency", "duration", "reopen_count",
                                  "vader_pos_comments", "vader_neg_comments", "vader_neut_comments",
                                  "csat"]
    csat_data = csat_data[list_of_features_and_label]

    #print("\nLoaded Data :\n------------------------------------")
    #print(csat_data.head())

    # print(csat_data[csat_data['csat']==1.0])
    #sys.exit()

    # Use custom label encoder
    clb = CustomLabelEncoder()

    # https://stackoverflow.com/questions/62005273/replacing-values-in-pandas-dataframe-using-replace-and-associated-warning
    csat_data = csat_data.copy()  # ?? in not, SettingWithCopyWarning: ehmm ..

    # df = df.replace(['old value'],'new value')
    csat_data['impact'] = csat_data['impact'].replace(['Critical'], clb.transform('impact', 'Critical'))
    csat_data['impact'] = csat_data['impact'].replace(['High'], clb.transform('impact', 'High'))
    csat_data['impact'] = csat_data['impact'].replace(['Medium'], clb.transform('impact', 'Medium'))
    csat_data['impact'] = csat_data['impact'].replace(['Low'], clb.transform('impact', 'Low'))

    csat_data['urgency'] = csat_data['urgency'].replace(['High'], clb.transform('urgency', 'High'))
    csat_data['urgency'] = csat_data['urgency'].replace(['Medium'], clb.transform('urgency', 'Medium'))
    csat_data['urgency'] = csat_data['urgency'].replace(['Low'], clb.transform('urgency', 'Low'))

    csat_data['csat'] = csat_data['csat'].replace([5.0], '5x')
    csat_data['csat'] = csat_data['csat'].replace([4.0], '4x')
    csat_data['csat'] = csat_data['csat'].replace([3.0], '3x')
    csat_data['csat'] = csat_data['csat'].replace([2.0], '2x')
    csat_data['csat'] = csat_data['csat'].replace([1.0], '1x')

    csat_data['csat'] = csat_data['csat'].replace(['5x'], clb.transform('csat', '5x'))
    csat_data['csat'] = csat_data['csat'].replace(['4x'], clb.transform('csat', '4x'))
    csat_data['csat'] = csat_data['csat'].replace(['3x'], clb.transform('csat', '3x'))
    csat_data['csat'] = csat_data['csat'].replace(['2x'], clb.transform('csat', '2x'))
    csat_data['csat'] = csat_data['csat'].replace(['1x'], clb.transform('csat', '1x'))

    # remove part of csat = 5
    # https://stackoverflow.com/questions/41170971/how-to-delete-fraction-of-rows-that-has-specific-attribute-value-from-pandas-dat

    # remove frac of csat = 5 (most of the available csat data) otherwise it'll skew the model towards csat = 5
    frac = params.getForProp('prop')['frac']
    csat_data = csat_data.drop(csat_data[csat_data['csat'] == 5].sample(frac=frac,random_state=0).index)

    # frac=23000
    # csat_data = csat_data.drop(csat_data[csat_data['csat'] == 4].head(frac).index)  # drop first frac records with csat 5

    # drop index column from csat_data
    #csat_data = csat_data.drop(['index'], axis=1)

    print(csat_data.head(5))
    print("-------------------------------")

    # change all values in column duration to float
    csat_data['duration'] = csat_data['duration'].astype(float)

    X_data = csat_data.drop(['csat'], axis=1)
    Y_data = csat_data['csat']


    #print(type(X_data))
    #print(type(Y_data))
    #sys.exit()

    # show unique values from Y_Data
    print(Y_data.unique())

    # Split training and test data
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data,test_size=0.2,random_state=0)

    print("\nTrain Test Dimensions:\n------------------------------------")
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)


    #from sklearn.utils.multiclass import type_of_target
    #y_type = type_of_target(Y_train)
    #print(y_type)

    #print(X_train.head(5))
    #sys.exit()


    # model fitting
    nesti = 1000
    regressor = LGBMClassifier(learning_rate=0.09,max_depth=-5,random_state=42,n_estimators=nesti).fit(X_train,Y_train,eval_set=[(X_test,Y_test),(X_train,Y_train)],
          verbose=20,eval_metric='logloss')

    print('Training accuracy {:.4f}'.format(regressor.score(X_train, Y_train)))
    print('Testing accuracy {:.4f}'.format(regressor.score(X_test, Y_test)))


    from shapash import SmartExplainer

    xpl = SmartExplainer(
        model=regressor
    )

    xpl.compile(x=X_test,
                y_target=Y_test # Optional: allows to display True Values vs Predicted Values
               )

    # https://github.com/MAIF/shapash/blob/master/tutorial/predictor_to_production/tuto-smartpredictor-introduction-to-SmartPredictor.ipynb
    predictor = xpl.to_smartpredictor()
    predictor.save('./LGBM_MODELS/predictor_lgbm1_csat5frac{}_nesti_{}.pkl'.format(frac,nesti))
    from shapash.utils.load_smartpredictor import load_smartpredictor
    predictor_load = load_smartpredictor('./LGBM_MODELS/predictor_lgbm1_csat5frac{}_nesti_{}.pkl'.format(frac,nesti))

    predictor_load.add_input(x=X_test)
    var = predictor_load.data["ypred"]

    print(var)

    app = xpl.run_app(title_story='Wiser CSAT (LGBM 1 model Explainer)', port=8021)





# EXECUTE MAIN
if __name__ == "__main__":
    main()

# END







