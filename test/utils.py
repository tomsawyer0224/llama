from cycler import V
import torch.nn as nn
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_metrics(log_dir: str):
    '''
    plots metrics from log directory that includes csv log file
    args:
        log_dir: path/to/log_dir
    returns:
        None
    '''
    dirs = os.listdir(log_dir)
    logfiles = [
        os.path.join(log_dir, d) for d in dirs if d.endswith('.csv')
    ]
    assert len(logfiles) == 1, \
    'log_dir should contain only one csv log file, please check'
    metrics = pd.read_csv(logfiles[0])

    org_columns = metrics.columns
    org_columns = [c for c in org_columns if 'test' not in c]
    epoch_columns = [c for c in org_columns if 'epoch' in c ]
    step_columns = [c for c in org_columns if 'step' in c ]
    other_columns = list(
        set(org_columns) - set(epoch_columns) - set(step_columns)
    ) + ['step']
    epoch_metrics = metrics[epoch_columns].set_index('epoch')
    step_metrics = metrics[step_columns].set_index('step')

    other_metrics = None
    if other_columns:
        other_metrics = metrics[other_columns].set_index('step')
        other_metrics_names = other_metrics.columns

    save_dir = os.path.join(log_dir, 'metric_curves')
    os.makedirs(save_dir, exist_ok = True)
    epoch_metrics_names = epoch_metrics.columns
    step_metrics_names = step_metrics.columns
    for name in epoch_metrics_names:
        fig, ax = plt.subplots(1,1)
        ax = epoch_metrics[name].dropna().plot(title = name, legend = True)
        fig.savefig(
            os.path.join(save_dir, name + '.png')
        )
        plt.close()
    for name in step_metrics_names:
        fig, ax = plt.subplots(1,1)
        ax = step_metrics[name].dropna().plot(title = name, legend = True)
        fig.savefig(
            os.path.join(save_dir, name + '.png')
        )
        plt.close()
    if other_metrics is not None:
        for name in other_metrics_names:
            fig, ax = plt.subplots(1,1)
            ax = other_metrics[name].dropna().plot(title = name, legend = True)
            fig.savefig(
                os.path.join(save_dir, name + '.png')
            )
        plt.close()
def save_image(
        images: list[torch.Tensor], 
        path: str, 
        titles: list[str] = None
    ) -> None:
    '''
    args:
        images: a list of 4D-tensors
        path: path to save images
        titles: title of tensors
    returns: None
    '''
    images = [
        make_grid(img).permute(1,2,0) for img in images
    ]
    n_samples = len(images)
    if titles:
        assert len(images) == len(titles)
    else:
        titles = ['']*n_samples
    fig, axes = plt.subplots(n_samples,1)
    for ax, img, tit in zip(axes, images, titles):
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(tit)
    fig.savefig(path,bbox_inches='tight')
    plt.close()

def plot_metrics_v2(metrics: dict[str,str], save_dir = None):
    '''
    plot metrics from multi csv files
    args:
        metrics: dict {
            "rope": "path/to/metrics1.csv",
            "abs": "path/to/metrics2.csv",
            ...
        }
        note that all csv files must be the same form (n_rows, n_cols)
        save_dir: directory to save results
    '''
    if save_dir is None:
        tmp = list(metrics.values())[0].split('/')
        save_dir = os.path.join(*tmp[:2],'metrics')
    #print(save_dir)
    df_metrics = {
        k:pd.read_csv(v) for k, v in metrics.items()
    }
    org_column_names = list(df_metrics.values())[0].columns
    index_columns = ['epoch', 'step']
    metric_names = [c for c in org_column_names if 'train' in c or 'val' in c]
    epoch_metrics = [c for c in metric_names if 'epoch' in c]
    step_metrics = [c for c in metric_names if 'step' in c]
    #print(index_columns)
    #print(metric_names)
    #print(epoch_metrics)
    #print(step_metrics)
    df_metrics = {
        k:v[index_columns+metric_names] for k, v in df_metrics.items()
    }
    #print(df_metrics['abs'][:5])
    df_metrics = {
        k:v.rename(
            columns = lambda c: f'{k}_{c}' if c not in index_columns else c
        )
        for k, v in df_metrics.items()
    }
    #print(df_metrics['rope'][:5])
    #print(df_metrics['abs'][:5])
    concat_df = pd.concat(df_metrics.values(), axis = 1)
    concat_df = concat_df.T.drop_duplicates().T
    #print(concat_df[:5].to_string())
    ''''''
    group_metrics = [
        [m for m in concat_df.columns if m.endswith(suf)] for suf in metric_names
    ]
    
    group_epoch_metrics = [
        [m for m in concat_df.columns if m.endswith(suf)] for suf in epoch_metrics
    ]
    group_step_metrics = [
        [m for m in concat_df.columns if m.endswith(suf)] for suf in step_metrics
    ]
    #print(group_metrics)
    #print(group_epoch_metrics)
    #print(group_step_metrics)
    os.makedirs(save_dir, exist_ok = True)
    for group in group_metrics:
        metric_name = group[0][group[0].index("_")+1:]
        if 'epoch' in metric_name:
            index_column = 'epoch'
        else: # step
            index_column = 'step'
        fname = os.path.join(save_dir, f'{metric_name}.png')
        fig, ax = plt.subplots(1,1)
        #print(group)
        #print(concat_df[[index_column]+group].dropna())
        #break
        '''
        ax = concat_df[[index_column]+group].dropna().plot(
                kind = 'line',
                title = metric_name,
                subplots = False,
                x = index_column,
                y = group
            )
        '''
        df = concat_df[[index_column]+group].dropna()
        cols = df.columns
        for c in cols[1:]:
            ax.plot(df[index_column], df[c], label = c)
        ax.legend()
        fig.suptitle(metric_name)
        fig.supxlabel(index_column)
        fig.supylabel(metric_name)
        fig.savefig(fname)
        plt.close()
    #print(f'saved all metrics to {save_dir}')




