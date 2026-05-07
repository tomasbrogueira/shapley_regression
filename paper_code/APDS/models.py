from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# from imblearn.pipeline import Pipeline as ImbPipeline
# from imblearn.under_sampling import RandomUnderSampler

from core.models.regression import ChoquisticRegression


RANDOM_STATE = 42


def get_models():
    """
    Models using class-weighting only (NO undersampling).
    """
    return {
        "Logistic_Regression": (
            Pipeline([
                ("clf", LogisticRegression(
                    max_iter=5000,
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                )) 
            ]), # change solver to solver="saga" and add penalty="none" for no_regularization
            {"clf__C": [1, 0.1, 0.01, 0.001, 1e-4, 1e-5]},
        ),

        "RandomForest": (
            Pipeline([
                ("clf", RandomForestClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ))
            ]),
            {
                "clf__n_estimators": [200, 500],
                "clf__max_depth": [5, 10, 15],
                "clf__min_samples_split": [5, 10, 20],
                "clf__min_samples_leaf": [2, 5, 10],
                "clf__max_features": ["sqrt", "log2"],
                "clf__ccp_alpha": [0.0, 0.001, 0.01, 0.1], #comment this line for no_regularization
            },
        ),

        "XGBoost": (
            Pipeline([
                ("clf", XGBClassifier(
                    use_label_encoder=False,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ))
            ]),
            {
                "clf__n_estimators": [200, 400],
                "clf__learning_rate": [0.01, 0.05],
                "clf__max_depth": [3, 4, 5],
                "clf__subsample": [0.7, 0.8, 0.9],
                "clf__colsample_bytree": [0.7, 0.8, 0.9],
                "clf__reg_lambda": [1, 5, 10, 20, 50], #comment this line for no_regularization
            },
        ),

        "MLP": (
            Pipeline([
                ("clf", MLPClassifier(random_state=RANDOM_STATE))
            ]),
            {
                "clf__solver": ["lbfgs"],
                "clf__hidden_layer_sizes": [(32,), (64, 32), (100,)],
                "clf__activation": ["relu", "tanh"],
                "clf__alpha": [0.0001, 0.01, 0.1, 1, 5], #add alpha=0.0 for no_regularization
                "clf__max_iter": [5000],
            },
        ),

        "Choquistic_Shapley": (
            Pipeline([
                ("clf", ChoquisticRegression(
                    representation="shapley",
                    scale_data=True,
                    k_add=2,
                    max_iter=5000,
                    solver="liblinear",
                    class_weight="balanced",
                )) 
            ]), # change solver to solver="saga" and add penalty="none" for no_regularization
            {"clf__C": [1, 0.1, 0.01, 0.001, 1e-4, 1e-5]},
        ),
    }


# ==========================================================
# --- UNDERSAMPLING MODELS (INTENTIONALLY COMMENTED) ---
# ==========================================================
# def get_undersampled_models():
#     undersampler = RandomUnderSampler(sampling_strategy=0.33, random_state=RANDOM_STATE)
#
#     return {
#     "Logistic_Regression_Undersampled": (
#         ImbPipeline([
#             ("under", undersampler),
#             ("clf", LogisticRegression(max_iter=5000, solver="liblinear", class_weight="balanced")) 
#         ]), # change solver to solver="saga" and add penalty="none" for no_regularization
#         {
#             "clf__C": [1, 0.1, 0.01, 0.001, 0.0001, 0.00001] 
#         }
#     ),

#     "RandomForest_Undersampled": (
#         ImbPipeline([
#             ("under", undersampler),
#             ("clf", RandomForestClassifier(
#                 class_weight="balanced", random_state=42, n_jobs=-1
#             ))
#         ]),
#         {
#             "clf__n_estimators": [200, 500],
#             "clf__max_depth": [5, 10, 15],
#             "clf__min_samples_split": [5, 10, 20],
#             "clf__min_samples_leaf": [2, 5, 10],
#             "clf__max_features": ["sqrt", "log2"],
#             "clf__ccp_alpha": [0.0, 0.001, 0.01, 0.1]  #comment this line for no_regularization
#         }
#     ),

#     "XGBoost_Undersampled": (
#         ImbPipeline([
#             ("under", undersampler),
#             ("clf", XGBClassifier(
#                 use_label_encoder=False,
#                 eval_metric="logloss",
#                 random_state=42,
#                 n_jobs=-1
#             ))
#         ]),
#         {
#             "clf__n_estimators": [200, 400],
#             "clf__learning_rate": [0.01, 0.05],
#             "clf__max_depth": [3, 4, 5],
#             "clf__subsample": [0.7, 0.8, 0.9],
#             "clf__colsample_bytree": [0.7, 0.8, 0.9],
#             "clf__reg_lambda": [1, 5, 10, 20, 50] #comment this line for no_regularization
#         }
#     ),

#     "MLP_Undersampled": (
#         ImbPipeline([
#             ("under", undersampler),
#             ("clf", MLPClassifier(
#                 solver='lbfgs',
#                 random_state=42
#             )) #add alpha=0.0 for no_regularization
#         ]),
#         {
#             "clf__hidden_layer_sizes": [(32,), (64,32), (100,)],
#             "clf__activation": ["relu", "tanh"],
#             "clf__alpha": [0.0001, 0.01, 0.1, 1.0],
#             "clf__max_iter": [5000]
#         }
#     ),

#     "Choquistic_Shapley_Undersampled": (
#         ImbPipeline([
#             ("under", undersampler),
#             ("clf", ChoquisticRegression(
#                 representation="shapley", scale_data=True, k_add=2,
#                 max_iter=5000, solver="liblinear", class_weight="balanced"
#             )) # change solver to solver="saga" and add penalty="none" for no_regularization
#         ]),
#         {
#             "clf__C": [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]  
#         }
#     )
#     }
