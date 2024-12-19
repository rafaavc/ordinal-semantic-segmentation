import os, json
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
alpha = 0.05

datasets_tweaks = {
    'Breast': {'rename': 'Breast Aesthetics'},
    'CervixMobileODT': {'rename': 'Cervix-MobileODT'},
    'Iris': {'rename': 'Mobbio'},
    'KelwinTeethISBI': {'rename': 'Teeth-ISBI'},
    'Teeth': {'rename': 'Teeth-UCV'},
    'BDDIntersected reduced1': {'rename': 'BDDIntersected reduced'},
    'BDDIntersected wroadagents1_noabstract': {'rename': 'BDDIntersected noabstract'},
    'BDDIntersected wroadagents1': {'ignore': True},
    'BDD10K wroadagents1_nodrivable': {'ignore': True},
    'BDD10K wroadagents1_nodrivable_noabstract': {'rename': 'BDD10K'},
    'Cityscapes': {},
}
loss_tweaks = {
    'CO2': {'rename': 'CE+O2'},
    'CO2_CSNP': {'rename': 'CE+O2+CSNP'},
    'ContactSurfaceDTv3': {'rename': 'CE+CSDT'},
    'ContactSurfaceTVv4': {'rename': 'CE+CSNP'},
    'CrossEntropy': {'rename': 'CE'},
    'CrossEntropyProbs': {'ignore': True},
    # not sure about these:
    'MultiCO2': {'rename': 'CE+O2'},
    'Multi_CO2_CSNP': {'rename': 'CE+O2+CSNP'},
    'MultiContactSurfaceDTv3': {'rename': 'CE+CSDT'},
    'MultiContactSurfaceTVv4': {'rename': 'CE+CSNP'},
}
metrics_tweaks = {
    'contact_surface': {'higher_better': False},
}
datasets_order = ['Breast Aesthetics', 'Cervix-MobileODT', 'Mobbio', 'Teeth-ISBI', 'Teeth-UCV', 'BDDIntersected reduced', 'BDDIntersected noabstract', 'BDD10K', 'Cityscapes']
loss_order = ['CE', 'CE+O2', 'CE+CSNP', 'CE+CSDT', 'CE+O2+CSNP']

# parse results
results = []
for path in ['results/biomedical', 'results/bdd100k']:
    for root, dirs, files in os.walk(path):
        if 'train.json' in files:
            train = json.load(open(os.path.join(root, 'train.json')))
            test = json.load(open(os.path.join(root, 'test.json')))
            dataset = train['dataset']
            if train['dataset_mask_type'] != 'na':
                dataset += ' ' + train['dataset_mask_type']
            if datasets_tweaks[dataset].get('ignore', False):
                continue
            if 'test_Cityscapes.json' in files:
                tests = [
                    (dataset, test),
                    ('Cityscapes', json.load(open(os.path.join(root, 'test_Cityscapes.json'))))
                ]
            else:
                tests = [(dataset, test)]
            for dataset, test in tests:
                for metric, values in test['0'].items():
                    if type(values) == float:
                        if train['loss'] in loss_tweaks and loss_tweaks[train['loss']].get('ignore', False):
                            continue
                        results.append({
                            'dataset': datasets_tweaks[dataset].get('rename', dataset),
                            'loss': loss_tweaks[train['loss']].get('rename', train['loss']),
                            'reg': train['regularization_weight'],
                            'metric': metric,
                            'results': [fold[metric] for fold in test.values()],
                        })
df = pd.DataFrame(results)
df.to_excel('df.xlsx')

# one-tailed paired t-test
def ttest(row):
    higher_better = True
    if row['metric'] in metrics_tweaks:
        higher_better = metrics_tweaks[row['metric']].get('higher_better', True)
    baseline = df[(df['dataset'] == row['dataset']) & (df['metric'] == row['metric']) & (df['loss'] == 'CE')]
    assert len(baseline) == 1
    t_stat, p_value = ttest_rel(row['results'], baseline['results'].iloc[0])
    if higher_better:
        return p_value/2 if t_stat > 0 else 1-p_value/2
    else:
        return p_value/2 if t_stat < 0 else 1-p_value/2

df['average'] = df['results'].apply(np.mean)
df['stdev'] = df['results'].apply(np.std)
df['pvalue'] = df.apply(ttest, 1)
df['significant'] = df['pvalue'] <= alpha

# choose best regularization according to Dice
for dataset in datasets_order:
    for loss in loss_order:
        dice_df = df[(df['dataset'] == dataset) & (df['loss'] == loss) & (df['metric'] == 'dice_coefficient_macro')]
        reg = dice_df.loc[dice_df['average'].idxmax()]['reg']
        df = df[(df['dataset'] != dataset) | (df['loss'] != loss) | (df['reg'] == reg)]

df.to_excel('df_reg.xlsx')

# latex
for metric in ['dice_coefficient_macro', 'percentage_of_unimodal_px', 'contact_surface']:
    higher_better = True
    if metric in metrics_tweaks:
        higher_better = metrics_tweaks[metric].get('higher_better', True)
    df2 = df[df['metric'] == metric].copy()
    # re-order
    df2['dataset'] = pd.Categorical(df2['dataset'], categories=datasets_order, ordered=True)
    df2['loss'] = pd.Categorical(df2['loss'], categories=loss_order, ordered=True)
    df2 = df2.sort_values(['dataset', 'loss'])
    # pivot table (rows=dataset, columns=loss)
    df2 = df2.pivot(columns='loss', index='dataset', values=['average', 'stdev', 'significant'])
    print(metric)
    print(r'\documentclass{standalone}')
    print(r'\begin{document}')
    print(r'\begin{tabular}{|l|' + 'r'*len(df2['average'].columns) + '|}')
    print(r'\hline')
    print('Dataset', *[c.replace('_', r'\_') for c in df2['average'].columns], sep=' & ', end=' \\\\\\hline\n')
    first_bdd = True
    for dataset, columns in df2.iterrows():
        if 'BDD' in dataset and first_bdd:
            first_bdd = False
            print(r'\hline')
        print(dataset, *['$' + (r'\mathbf{' if bold else '') + f'{avg*100:.1f}\\pm{std*100:.1f}' + ('}' if bold else '') + '$' for avg, std, bold in zip(columns['average'], columns['stdev'], columns['significant'])], sep=' & ', end=' \\\\\n')
    print(r'\hline')
    print(r'\end{tabular}')
    print(r'\end{document}')
