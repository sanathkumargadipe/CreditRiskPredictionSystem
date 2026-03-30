from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def build_model():

    lgbm = LGBMClassifier()

    cat = CatBoostClassifier(verbose=0)

    svm = SVC(probability=True)

    stack = StackingClassifier(

        estimators=[
            ('lgbm', lgbm),
            ('cat', cat),
            ('svm', svm)
        ],

        final_estimator=LogisticRegression()

    )

    return stack