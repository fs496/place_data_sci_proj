"""
Analyze results from testing the dependence of gradient descent on step size.
"""
import pandas as pd
import seaborn.objects as so
import seaborn as sns
import numpy as np
import itertools


DATA_FOLDER = "(TOP LEVEL FOLDER REMOVED FOR PRIVACY)"
SAVE_FOLDER = "(TOP LEVEL FOLDER REMOVED FOR PRIVACY)/place_data_sci_proj/figures"


if __name__ == '__main__':
    # READ DATA
    df = pd.read_pickle(f"{DATA_FOLDER}/grad_desc_rel_True_stop_on_loss.pkl")
    df['e_cat'] = pd.Categorical(df.e)
    
    # UNIVARIATE PLOTS: E ONLY
    # Convergence rate by e
    df_conv = df.groupby('e_cat').converged.sum() / df.groupby('e_cat').size() * 100
    df_conv = df_conv.reset_index().rename(columns={0: 'convergence_rate'})
    
    fig_conv = (
        so.Plot(df_conv, x='e_cat', y='convergence_rate')
        .add(so.Bar())
        .label(title = "Convergence rate by step size")
        .label(x='Step size (e)', y='Convergence rate (%)')
    )
    fig_conv.save(f'{SAVE_FOLDER}/fig_conv.png', dpi=300, bbox_inches='tight')
    
    # Number of steps by e
    ax_steps = sns.boxplot(
        data=df.loc[df.converged], x='e_cat', y='num_steps', log_scale=10
    )
    ax_steps.set(
        xlabel='Step size (e)',
        ylabel='Number of steps',
        title='Number of steps taken for converged runs by step size'
    )
    fig_steps = ax_steps.get_figure()
    fig_steps.savefig(
        f'{SAVE_FOLDER}/fig_steps.png',
        format='png', dpi=300, bbox_inches='tight'
    )
    
    # Percent error in b by e
    df['error_b_pct'] = df['error_b'] * 100
    ax_err = sns.boxplot(
        data=df.loc[df.converged], x='e_cat', y='error_b_pct', log_scale=10
    )
    ax_err.set(
        xlabel='Step size (e)',
        ylabel='Percent error in b (%)',
        title='Percent error in b for converged runs by step size'
    )
    fig_err = ax_err.get_figure()
    fig_err.savefig(
        f'{SAVE_FOLDER}/fig_err.png',
        format='png', dpi=300, bbox_inches='tight'
    )
    
    # BIVARIATE PLOTS
    # Calculate ||x||, ||y||, ||x||/||y||, theta, and error of b0
    df['x_len'] = df['x'].apply(np.linalg.norm)
    df['y_len'] = df['y'].apply(np.linalg.norm)
    df['x_y_ratio'] = df['x_len'] / df['y_len']
    df['xy_angle'] = df.apply(
        lambda d: np.rad2deg(np.arccos(
            np.dot(d['x'], d['y']) / (d['x_len'] * d['y_len'])
        )),
        axis=1
    )
    df['rel_err_b0'] = (df['b0'] - df['b_true']) / df['b_true']
    # Bin these values
    df['x_len_bin'] = pd.cut(df.x_len, bins=np.arange(0, 5.5, 0.5))
    df['y_len_bin'] = pd.cut(df.y_len, bins=np.arange(0, 5.5, 0.5))
    df['x_y_ratio_bin'] = pd.cut(df.x_y_ratio, bins=np.arange(0, 10, 0.5))
    df['xy_ang_bin'] = pd.cut(df.xy_angle, bins=np.arange(0, 200, 20))
    df['err_b0_bin'] = pd.qcut(df.rel_err_b0.round(2), q=10)

    df.loc[~df.converged, 'num_steps'] = np.nan
    df.loc[~df.converged, 'error_b_pct'] = np.nan
    
    # Generate heatmap for each combination of performance metric
    # and second variable in addition to e
    index_cols = [
        ['x_len_bin', 'Length of x'],
        ['y_len_bin', 'Length of y'],
        ['x_y_ratio_bin', 'Ratio of lengths of x and y'],
        ['xy_ang_bin', 'Angle between x and y (degrees)'],
        ['err_b0_bin', 'Percent error in b0']
    ]
    value_cols = [
        ['converged', 'Convergence rate (percent)'],
        ['num_steps', 'Median number of steps per run'],
        ['error_b_pct', 'Percent error in estimated b']
    ]
    for index_col, value_col in itertools.product(index_cols, value_cols):
        column_col = ['e_cat', 'Step size (e)']

        heat_data = df.groupby(['e_cat', index_col[0]]).agg({
            'converged': lambda x: x.sum() / x.count() * 100,
            'num_steps': lambda x: x.median(skipna=True),
            'error_b_pct': lambda x: x.median(skipna=True),
            'id': lambda x: x.count()
        })\
            .rename(columns={'id': 'num_runs'})\
            .fillna({'num_runs': 0})\
            .reset_index()

        ax_heat = sns.heatmap(heat_data.pivot(
            index=index_col[0], columns=column_col[0], values=value_col[0]
        ))
        title = f'{value_col[1]} by {column_col[1]} and {index_col[1]}'
        ax_heat.set(
            xlabel=column_col[1],
            ylabel=index_col[1],
            title=title
        )
        fig_heat = ax_heat.get_figure()
        fig_heat.savefig(
            f'{SAVE_FOLDER}/{title}.png',
            format='png', dpi=300, bbox_inches='tight'
        )
        ax_heat = ax_heat.clear()
        fig_heat = fig_heat.clear()
        del ax_heat
        del fig_heat
