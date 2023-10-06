from datetime import datetime
from pathlib import Path
import pandas as pd
import patchworklib as pw
from plotnine import *
import janitor


def plot_cat_features(dataframe: pd.DataFrame) -> tuple:
    """
    Plots the categorical features of the data.
    """
    # create fill color for plot #06092c , #f1eacb
    fill_color = ['#5c86ff', '#e33d75']

    # order of x-axis
    x_axis_order = ['low', 'medium', 'high']

    # select object features
    d1 = dataframe[['department', 'salary', 'exit']]

    # prepare data for visualization
    d2_melt = pd.melt(pd.crosstab(d1.department, [d1.salary, d1.exit]))
    d3_melt = pd.melt(pd.crosstab(d1.exit, [d1.department, d1.salary]),
                      ignore_index=False).reset_index()

    # plot 1: Count salary category with exit as fill
    plot_salary_exit = ggplot(d2_melt) \
                       + geom_col(aes(x='salary', y='value', fill='factor(exit)')) \
                       + scale_fill_manual(values=fill_color, labels=['No', 'Yes']) \
                       + scale_x_discrete(limits=x_axis_order) \
                       + labs(
        fill='Exit',
        title='How many employees left the company categorized by salary?',
        y='Count') \
                       + theme_minimal()

    # plot 2: Count salary category by department with exit as fill
    plot_department_exit = ggplot(d3_melt) \
                           + geom_col(aes(x='salary', y='value', fill='factor(exit)')) \
                           + scale_fill_manual(values=fill_color, labels=['No', 'Yes']) \
                           + scale_x_discrete(limits=x_axis_order, ) \
                           + facet_grid(facets='.~department') \
                           + labs(
        fill='Exit',
        title='What is the distribution of exits per department by salary?',
        y='Count') \
                           + theme_minimal() \
                           + theme(figure_size=(12, 5), axis_text_x=element_text(rotation=80, hjust=1, size=8))

    return (plot_salary_exit, plot_department_exit)


def plot_num_float_features(float_feat_df, pallette):
    """

    """
    satisfaction_level_kde = ggplot(float_feat_df, aes()) \
                             + geom_density(aes(x='satisfaction_level', fill='exit'), alpha=0.3) \
                             + scale_fill_manual(values=pallette, labels=['No', 'Yes']) \
                             + theme_minimal() \
                             + labs(title='Satisfaction Level Density')

    last_evaluation_kde = ggplot(float_feat_df, aes()) \
                          + geom_density(aes(x='last_evaluation', fill='exit'), alpha=0.3) \
                          + scale_fill_manual(values=pallette, labels=['No', 'Yes']) \
                          + annotate(geom_hline, yintercept=(3.1, 2.45)) \
                          + annotate(geom_rect, ymin=2.45, ymax=3.1, xmin=0.3, xmax=float('inf'), alpha=0.1) \
                          + theme_minimal() \
                          + labs(title="Performance Evaluation Density")

    kde_plot = [satisfaction_level_kde, last_evaluation_kde]
    plot_g = [pw.load_ggplot(p, figsize=(8, 4)) for p in kde_plot]
    plot_mat_g = (plot_g[0] | plot_g[1])

    return plot_mat_g


def plot_discrete_features(df, pallette):
    """
    Plot discrete features of the dataset
    """
    int_names = df.columns.to_list()
    hist_plot_list = []
    hist_title_list = []
    for names in int_names:
        p = ggplot(df) + geom_histogram(aes(x=names, fill='factor(exit)')) + facet_grid('.~exit') \
            + scale_fill_manual(label=['No', 'Yes'], values=pallette) + theme_minimal()
        hist_plot_list.append(p)

    pw_plot = [pw.load_ggplot(plot, figsize=(5, 3)) for plot in hist_plot_list]
    pw_plot_g = (pw_plot[0] | pw_plot[1] | pw_plot[2])

    return pw_plot_g


def plot_binary_features_promotion(int_features, pallette):
    """
    Plot binary features
    """
    plot_features = ['number_project', 'time_spend_company', 'average_monthly_hours']
    plot_title = ['', '']

    gg_list = []
    for features in plot_features:
        gg = ggplot(int_features) \
             + geom_histogram(aes(x=features, fill='factor(exit)')) \
             + facet_grid('.~promotion_last_5years') \
             + scale_fill_manual(label=['No', 'Yes'], values=pallette) \
             + theme_minimal()
        gg_list.append(gg)

    plot_g = [pw.load_ggplot(plot, figsize=(5, 3)) for plot in gg_list]
    plot_mat_g = (plot_g[0] | plot_g[1] | plot_g[2])

    return plot_mat_g


def plot_binary_features_accident(int_features, pallette):
    """

    """
    plot_features = ['number_project', 'time_spend_company', 'average_monthly_hours']

    gg_list = []
    for features in plot_features:
        gg = ggplot(int_features) \
             + geom_histogram(aes(x=features, fill='factor(exit)')) \
             + facet_grid('.~work_accident') \
             + scale_fill_manual(label=['No', 'Yes'], values=pallette) \
             + theme_minimal()
        gg_list.append(gg)

    plot_g = [pw.load_ggplot(plot, figsize=(5, 3)) for plot in gg_list]
    plot_mat_g = (plot_g[0] | plot_g[1] | plot_g[2])

    return plot_mat_g


# cleaning functions
def compare(new: pd.DataFrame, original: pd.DataFrame):
    datetime_stamp = datetime.now().strftime(' %A : %d/%m')
    print(
        f"""
    {datetime_stamp:=>50}
    Shape
    {'-' * 50}
    original_df : {original.shape}
    dedup df : {new.shape}

    Size
    {'-' * 50}
    original_df : {original.size}
    dedup df : {new.size}

    Memory Usage
    {'-' * 50}
    original_df : {original.memory_usage(deep=True)}
    {'~' * 50}
    dedup df : {new.memory_usage(deep=True)}

    Memory Usage Sum
    {'-' * 50}
    original_df : {original.memory_usage(deep=True).sum()}
    {'~' * 50}
    dedup df : {new.memory_usage(deep=True).sum()}
        """
    )
