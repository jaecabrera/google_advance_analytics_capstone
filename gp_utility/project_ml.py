from collections.abc import Iterator
from typing import List

import numpy as np
import pandas as pd
from plotnine import *
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import silhouette_score, f1_score, classification_report

M_RN_STATE = 5


def kmeans_inertia(num_clusters, x_vals):
    """
    :param num_clusters:
    :param x_vals:
    :return:
    """
    inertia = []
    for num in num_clusters:
        kms = KMeans(n_clusters=num, random_state=M_RN_STATE, n_init=10)
        kms.fit(x_vals)
        inertia.append(kms.inertia_)

    return inertia


def kmeans_sil(num_clusters, x_vals):
    """
    :param num_clusters:
    :param x_vals:
    :return:
    """
    sil_score = []
    for num in num_clusters:
        kms = KMeans(n_clusters=num, random_state=M_RN_STATE, n_init=10)
        kms.fit(x_vals)
        sil_score.append(silhouette_score(x_vals, kms.labels_))

    return sil_score


def plot_kmean_feat(x_feat, scaled_important_features):
    """

    :param x_feat:
    :return:
    """
    values = ['#5c86ff', '#db0049', "#EBD323"]
    g_plot = (ggplot(scaled_important_features) \
              + geom_point(aes(x=x_feat, y='satisfaction_level', fill='factor(clusters)'), alpha=0.3) \
              + scale_fill_manual(values=values, labels=[0, 1, 2]) \
              + theme_minimal())
    return g_plot


def save_kmean_3_cluster_plots(scaled_important_features):
    """
    :return:
    """
    x_features = ['last_evaluation', 'average_monthly_hours', 'number_project', 'time_spend_company']
    values = ['#FFA500', '#9467bd', '#008080']

    for feat in x_features:
        g = ggplot(scaled_important_features) \
            + geom_point(aes(x=feat, y='satisfaction_level', fill='factor(clusters)'), alpha=0.3) \
            + scale_fill_manual(values=values, labels=[0, 1, 2]) \
            + theme_minimal()

        yield g


def get_table_feature_importance(m_feature_importance) -> List:
    importance_table_list = []
    for key, value in m_feature_importance.items():
        importance_table = value.T
        importance_table.columns = ['importance']
        importance_table = importance_table \
            .sort_values(by='importance', ascending=False) \
            .reset_index()

        importance_table.columns = ['feature', 'importance']
        importance_table_list.append(importance_table)

    return importance_table_list


def plot_feature_importance(m_feature_importance) -> Iterator[ggplot]:
    importance_table_list = get_table_feature_importance()
    for key, value in m_feature_importance.items():
        importance_table = value.T
        importance_table.columns = ['importance']
        importance_table = importance_table \
            .sort_values(by='importance', ascending=False) \
            .reset_index()

        importance_table.columns = ['feature', 'importance']
        importance_table_list.append(importance_table)

        plots = ggplot(importance_table, aes(x='reorder(feature, -importance)', y='importance')) \
                + ggtitle(key) \
                + geom_col() \
                + theme_gray() \
                + theme(axis_text_x=element_text(rotation=90, hjust=1, size=8),
                        figure_size=(10, 5), dpi=75)

        yield plots


# create an empty feature importance dict

def train_models(model_: dict, split: List, m_feature_importance: dict, feature_names):
    """
    :description:
        Fit and train data to evaluate accuracy scores and clas
    :model: A dictionary of sklearn classification models
    :model type: dict
    :split: A training and testing split
    :split type: list
    """

    def print_f1_score(model_name, f1_score_val):
        return print(f"""
        Model: {model_name} {'-' * 20}
        f1-score: {np.round(f1_score_val * 100, 2)}""")

    def fit_data(model_name, model, m_split, all_feature_names):

        m_ = model.fit(m_split[0], m_split[2])
        m_pred = m_.predict(m_split[1])
        f1_score_val = f1_score(m_split[3], m_pred, zero_division=0)
        print_f1_score(model_name, f1_score_val)
        print(classification_report(m_split[3], m_pred))

        if model_name in ('DecisionTreeClassifier', 'RandomForestClassifier'):
            fi = pd.DataFrame(m_.feature_importances_).T
            fi.columns = all_feature_names
            m_feature_importance[model_name] = fi

        if model_name in ('SupportVectorClassifier', 'LogisticRegression'):
            try:
                coef = pd.DataFrame(m_.coef_)
                coef.columns = all_feature_names
                m_feature_importance[model_name] = coef
            except Exception as e:
                print(e)
                pass

    # train iterate all models in dict
    for model_name, model in model_.items():

        if model_name is 'SupportVectorClassifier' or 'LogisticRegression':
            fit_data(model_name, model, split, feature_names)

        else:
            fit_data(model_name, model, split, feature_names)


def get_roc_auc(y_test: pd.Series, model_name: str, split: List, score_dict: dict,
                classification_model) -> pd.DataFrame:
    """
    :param classification_model: Dict that contains model name and Model
    :param str model_name: String of model names to be used.
    :param list split: Split data from train_test_split().
    :param dict score_dict: Empty dictionary with model names.
    :returns: Pandas DataFrame for Plotting.
    """
    model = classification_model[model_name].fit(split[0], split[2])
    model_pred = model.predict_proba(split[1])[:, 1]
    y_test_array = y_test.to_numpy()

    # get auc scores
    false_pos_r, true_pos_r, thresh = roc_curve(y_test, model_pred, pos_label=1)
    auc_roc_df = pd.DataFrame([false_pos_r, true_pos_r]).T
    auc_roc_df.columns = ['fpr', 'tpr']
    auc_roc_df['model_name'] = model_name
    score_dict[model_name] = roc_auc_score(y_test_array, model_pred)

    return auc_roc_df
