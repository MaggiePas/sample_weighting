from typing import List, Any

from load_data import *
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
from datasets import *
from models import *
from train_test_utils import *
from dataframe_utils import *
from tabulate import tabulate
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.stats import pearsonr, ttest_ind
warnings.filterwarnings('ignore')

run_with_seed = 1964 #random.randint(0, 4294967295)
torch.manual_seed(run_with_seed)
np.random.seed(run_with_seed)
torch.cuda.manual_seed(run_with_seed)

# Load data
path = './source_data/lab_data/combined_training_data_with_trauma_vars_longitudinal_07_26.csv'

# Path to save the processed version of the dataset, if needed
w_path = ''

input_feats = load_longitudinal_tabular_data(input_path=path, write_path=w_path, quick=True, write_csv=False)

# We are predicting age and construct in a multi-task setting
trauma = False
plot_flag = True
# seq2seq dictates if we predict all time-steps or only the last one
seq2seq = False
batch_size = 25
additional_val = False
# mode 1 is the model without the regularization term on the eigenvalues
mode = 1
# ablation study for number of neighbors
knn = 30
# number of eigenbasis vectors to use
k = 2 #13
max_epochs = 100
weight_lr = 1e-5
model_lr = 1e-4
negative_alpha = 1
alpha = 0.001
centering = 0.9
plot_val = True
plot_train = True
construct = 'demo_diag'

# preprocessing for spectral method implementation
basis = read_csv(f'./lab_data_trauma_aud/eigenvectors_aud_trauma_lower_pert_knn_{knn}.csv', separator=',')
# basis = read_csv('./eigenvectors.csv', separator=',')
basis = basis.loc[basis['subjects'].isin(np.unique(input_feats['subject']))]
eigenvalues = pd.read_csv(f'./lab_data_trauma_aud/eigenvalue_aud_trauma_lower_pert_knn_{knn}.txt', sep='\r\n', header=None)
eigenvalues = torch.tensor(eigenvalues.values, dtype=torch.float32)[:k]


# ================================================================================================================
labels_train = {}
labels_test = {}
labels_control_diseased = {}
traumas = {}

for key in list(np.unique(input_feats.loc[:, 'subject'])):
    subj_v = input_feats[input_feats['subject'] == key]
    labels_train[key] = subj_v.loc[:, construct]
    labels_test[key] = subj_v.loc[:, construct]
    traumas[key] = subj_v.loc[:, 'trauma']
    
split_stratified_labels = []
for key in list(np.unique(input_feats.loc[:, 'subject'])):
    split_stratified_labels.append(labels_train[key].max())

# We are doing 5-fold cross validation se we are creating dictionaries to save the data for each fold
partition = {}
folds = {}
subject_folds = {}
mfb_folds = {}
folds_of_the_labels_train = {}
folds_of_the_labels_test = {}
folds_of_the_traumas = {}
counter = 0

# We are creating the object scaler to normalize our input data from -1 to 1
scaler = MinMaxScaler(feature_range=(-1,1))

# We are performing stratified CV since we hace class imbalance
kf = StratifiedKFold(n_splits=5, shuffle=False)  # False for reproducible folds
X = np.array(list(np.unique(input_feats.loc[:, 'subject'])))
y = np.array(split_stratified_labels)
kf.get_n_splits(X, y)

input_feats = input_feats.drop(columns=['trauma'])

# shap values of all subjects across folds
shap_values_deep = []

# We want to separate the data to train and test sets for each fold, normalize them and also 
# exclude the 'fake control' subjects from the test set. 
# 'Fake controls' are subjects that have 'True' in any depression construct in at least one of the visits. 
# The four constructs are: 'positive_valence', 'negative_valence', 'arousal', 'cognitive' 
for train_index, test_index in kf.split(X, y):
    labels_train_fold = labels_train.copy()
    labels_test_fold = labels_test.copy()
    traumas_fold = traumas.copy()
    subject_traumas = {}
    subject_post = {}
    train_subj = input_feats.loc[input_feats['subject'].isin(X[train_index])]
    y_train = train_subj.loc[:, construct]
    test_subj = input_feats.loc[input_feats['subject'].isin(X[test_index])]
    train_subj = train_subj.drop(columns='demo_diag')
    test_subj = test_subj.drop(columns='demo_diag')

    # In this version of the dataset the 21 first columns contain features that should not be used as input 
    # for the prediction.
    # These features are the personal information, like sex, race etc., the labels that we are predicting and other
    # potential prediction variables.
    X_train = train_subj.iloc[:, 2:]
    X_train = scaler.fit_transform(X_train)
    
    X_train = pd.DataFrame(data=X_train, columns=train_subj.columns[2:])
    X_train = X_train.set_index(train_subj.index)
    X_train.insert(0, 'subject', train_subj.loc[:, 'subject'], True)

    X_test = test_subj.iloc[:, 2:]
    X_test = scaler.transform(X_test)
    
    X_test = pd.DataFrame(data=X_test, columns=test_subj.columns[2:])
    X_test = X_test.set_index(test_subj.index)
    X_test.insert(0, 'subject', test_subj.loc[:, 'subject'], True)

    partition['test'] = list()
    partition['train'] = list()
    for subject in input_feats.subject.unique():
        if subject in list(X[train_index]):
            subj_visits = X_train[X_train['subject'] == subject]
            subject_traumas[subject] = subj_visits
            partition['train'].append(subject)
            
        elif subject in list(X[test_index]):
            subj_visits = X_test[X_test['subject'] == subject]
            subject_traumas[subject] = subj_visits
            partition['test'].append(subject)
    folds[counter] = partition.copy()

    # Subject-specific dataset with all the visits and post-processed for training
    for key in list(partition['train'] + partition['test']):
        df = subject_traumas[key]
        df = df.iloc[:, 1:]
        subject_post[key] = df

    subject_folds[counter] = subject_post.copy()
    folds_of_the_labels_train[counter] = labels_train_fold.copy()
    folds_of_the_labels_test[counter] = labels_test_fold.copy()
    folds_of_the_traumas[counter] = traumas_fold.copy()

    # Since we have class imbalance we are using weights in the binary cross entropy during training
    # These weights are calculated separately for each fold since we only take the training 
    # set into account to calculate them.
    number_neg_samples = np.sum(y_train.values == False)
    num_pos_samples = np.sum(y_train.values == True)
    mfb = number_neg_samples / num_pos_samples
    mfb_folds[counter] = mfb.copy()

    counter += 1

# Parameters
params = {'shuffle': True,
          'num_workers': 0,
          'batch_size': 1} # One batch contains all the visits of each subject

results = {}
results_additional = {}

# Train models for each fold
plot_fold_dict = {}
plot_fold_train_weight = {}
for fold in folds.keys():

    # Get the data per fold
    partition = folds[fold]
    subject_post = subject_folds[fold]
    pos_weight = mfb_folds[fold]
    labels_train_f = folds_of_the_labels_train[fold]
    labels_test_f = folds_of_the_labels_test[fold]
    traumas_f = folds_of_the_traumas[fold]

    # spectral related data
    train_basis = basis.loc[basis['subjects'].isin(partition['train'])]
    val_basis = basis.loc[basis['subjects'].isin(partition['test'])]
    lookup_dictionary = {}
    lookup_dictionary_val = {}
    for i in range(len(np.unique(train_basis['subjects']))):
        lookup_dictionary[np.unique(train_basis['subjects'])[i]] = i
    for i in range(len(np.unique(val_basis['subjects']))):
        lookup_dictionary_val[np.unique(val_basis['subjects'])[i]] = i
    train_basis = train_basis.drop(columns=['subjects'])
    val_basis = val_basis.drop(columns=['subjects'])
    train_basis = torch.tensor(train_basis.values, dtype=torch.float32)[:, :k]
    val_basis = torch.tensor(val_basis.values, dtype=torch.float32)[:, :k]

    basis_scaler = StandardScaler()
    scaler.fit(train_basis)
    scaler.transform(val_basis)


    # Dataset generators
    training_set = Dataset(partition['train'], subject_post, labels_train_f, traumas_f, trauma)
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    validation_set = Dataset(partition['test'], subject_post, labels_test_f, traumas_f, trauma)
    validation_generator = torch.utils.data.DataLoader(validation_set, **params)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # The parameters I trained with, after hyperparameter tuning
    epoch = 0
    feature_dim = 128
    input_dim = next(iter(training_generator))[0].shape[2]
    output_dim = 1
    n_layers = 1
    hidden_dim = 64

    model = GRUNet(feature_dim=feature_dim, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                          n_layers=n_layers, seq2seq=seq2seq, device=device, drop_prob=0.0)
    # Loss chosen for binary classification task (not acutally used here for spectral decomposition model)
    score_criterion = nn.BCEWithLogitsLoss()
    # score_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).float().to(device))
    criterion = score_criterion
    weight = torch.rand(k, requires_grad=True)
    optimizer = torch.optim.Adam([ {'params': model.parameters()}, {'params': weight, 'lr': weight_lr}], lr=model_lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.5)
    model.to(device)
    model.train()

    model_trained, h, plot, hist_data, weight, shap_per_fold = train_gru_spectral(model=model, criterion=criterion, optimizer=optimizer, max_epochs=max_epochs,
                                         train_loader=training_generator, val_loader=validation_generator,
                                         device=device, seq2seq=seq2seq, params=params, batch_size=batch_size, scheduler=scheduler, 
                                         weight=weight, lookup=lookup_dictionary, lookup_val=lookup_dictionary_val, basis=train_basis, basis_val=val_basis, eigenvalues = eigenvalues, mode=mode,
                                         negative_alpha=negative_alpha, alpha=alpha, centering=centering)
    for SHAPs in shap_per_fold:
        shap_values_deep.append(SHAPs)

    # more visualization of subjects of different weights, let's only do this for fold 0 for now
    data_lookup = read_csv('./lab_data_trauma_aud/graph_factors_raw_lab_data.csv', separator=',')
    if plot_val:
        data = np.zeros((len(partition['test']), 3))
        c = np.zeros((len(partition['test']), ))
        for i, subject in enumerate(partition['test']):
            data[i, :] = data_lookup[data_lookup['subject'] == subject][['demo_sex', 'age', 'demo_ses']].values
            c[i] = centering + weight @ val_basis[lookup_dictionary_val[subject]]
    else:
        data = np.zeros((len(partition['train']), 3))
        c = np.zeros((len(partition['train']), ))
        for i, subject in enumerate(partition['train']):
            data[i, :] = data_lookup[data_lookup['subject'] == subject][['demo_sex', 'age', 'demo_ses']].values
            c[i] = centering + weight @ train_basis[lookup_dictionary[subject]]
            if subject not in plot_fold_train_weight.keys():
                plot_fold_train_weight[subject] = [c[i]]
            else:
                plot_fold_train_weight[subject].append(c[i])
    x, y, z = data[:, 0], data[:, 1], data[:, 2]  # for show
    plot_fold_dict[fold] = (x, y, z, c)

    if fold == 4:
        if not plot_val:
            data_lookup = read_csv(f'./lab_data_aud_trauma/graph_factors_raw_lab_data.csv', separator=',')
            overall_data = np.zeros((len(plot_fold_train_weight), 3))
            overall_c = np.zeros((len(plot_fold_train_weight), ))
            i = 0
            output = []
            for subject in plot_fold_train_weight:
                overall_c[i] = torch.mean(torch.tensor(plot_fold_train_weight[subject])).item()
                overall_data[i] = data_lookup[data_lookup['subject'] == subject][['sex', 'age', 'ses']].values
                output.append([subject, overall_c[i]])
                i += 1
            print(ttest_ind(overall_c[np.where(overall_data[:, 0] == 0.)], overall_c[np.where(overall_data[:, 0] == 1.)]))
            print(pearsonr(overall_c, overall_data[:, 1]))
            print(pearsonr(overall_c, overall_data[:, 2]))
            output = pd.DataFrame(output, columns=['subject', 'weight_train'])
            output.to_csv(f'./analysis_lab_data/weight_train_knn_{knn}_eigenbasis_{k}_centering_{centering}_lab_data.csv')
        else:
            concat_x = []
            concat_y = []
            concat_z = []
            concat_c = []
            for fold in range(5):
                concat_x.append(plot_fold_dict[fold][0])
                concat_y.append(plot_fold_dict[fold][1])
                concat_z.append(plot_fold_dict[fold][2])
                concat_c.append(plot_fold_dict[fold][3])
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            x = np.concatenate(concat_x, axis=0)
            y = np.concatenate(concat_y, axis=0)
            z = np.concatenate(concat_z, axis=0)
            c = np.concatenate(concat_c, axis=0)
            sorted_c = np.sort(c)
            cutoff = []
            for i in range(3):
                cutoff.append(sorted_c[i * (len(c) // 3)])
            cutoff.append(sorted_c[-1])
            new_c = np.zeros((len(c), ))
            for i in range(len(c)):
                for j in range(len(cutoff)-1):
                    if c[i] >= cutoff[j] and c[i] < cutoff[j+1]:
                        new_c[i] = j
                        break
            p = ax.scatter(x, y, z, c=new_c, cmap=plt.cm.viridis)
            ax.set_xlabel('sex')
            ax.set_ylabel('age')
            ax.set_zlabel('ses')

            ax.set_box_aspect([np.ptp(i) for i in data.T])  # equal aspect ratio
            cbaxes = fig.add_axes([0.1, 0.1, 0.03, 0.8]) 
            plt.colorbar(p, cax=cbaxes)
            fig.savefig(f"./plots_lab_data/weights_3d_all_knn_{knn}_lab_data")
            plt.clf()
            # fig, ax = plt.subplots()
            # ax.hist(c, bins=20, color='maroon')
            # ax.set(xlabel='Weights', ylabel='Counts', title='Weight W Distribution Fold All'.format(fold))
            # fig.savefig("./plots/hist_all.png")
            plt.rcParams['figure.dpi'] = 500
            plt.rcParams['savefig.dpi'] = 500

            # plt.xlim(0.62, 0.78)
            plt.xlim(0.7, 1.1) # edit accordingly
            sns.set(font_scale = 1.8)
            sns.set_style("whitegrid")
            sns.set()
            sns.despine(left=True) #plum, navajowhite, skyblue https://matplotlib.org/stable/gallery/color/named_colors.html
            ax = sns.histplot(data=c, bins=15, color='plum').set_title(f'Sample Weights Lab Data')
            plt.ylabel("Number of Subjects")
            plt.xlabel(f"Sample Weight Distribution Lab Data Knn: {knn}")
            plt.savefig(f"./plots_lab_data/hist_all_knn_{knn}_lab_data")
            np.save('histogram_data_lab_data', c)

            print(ttest_ind(c[np.where(x == 0.)], c[np.where(x == 1.)]))
            print(pearsonr(c, y))
            print(pearsonr(c, z))

            subjects = []
            for i in range(5):
                subjects += folds[i]['test']
            output = pd.DataFrame(zip(subjects, c), columns=['subject', 'weight_val'])
            print(len(output))
            output.to_csv(f'./analysis_lab_data/weight_val_knn_{knn}_eigenbasis_{k}_centering_{centering}_lad_data.csv')

    print(f'Model training complete for fold {fold} complete.')
    # Uncomment to save models
    #torch.save(model_trained.state_dict(), output_path)
    #print(f'Model Saved at: {output_path}')
    
    if seq2seq:
        results[f'split{fold}'] = evaluate_all_timesteps_per_subject(model=model_trained, val_loader=validation_generator, hidden=h, device=device)
    else:
        results[f'split{fold}'] = evaluate_last_timestep(model=model_trained, val_loader=validation_generator, device=device, criterion=criterion, lookup=lookup_dictionary_val, weight=weight, basis=val_basis, centering=centering)


df_gradients = pd.DataFrame(shap_values_deep, columns=df.columns)
# Compute average for each row
df_gradients['average'] = df_gradients.mean(axis=0)
# Average gradients by column across subjects
average_df = pd.DataFrame(df_gradients.mean(axis=0)).T
# Sort columns by 'average' to find the max feature
df_sorted = df_gradients.sort_values(by='average', ascending=False) # Use ascending=True for ascending order
sorted_df = average_df.sort_values(by=0, axis=1, ascending=False)
df_T = sorted_df.T.reset_index()
# Keep the 15 largest columns
top_df_T = df_T.nlargest(15, 0)

df_T.to_csv(f'./analysis_lab_data/feature_ablation_{knn}_eigenbasis_{k}_centering_{centering}_lab_data.csv')

plt.figure(figsize=(12, 10))
plt.title(f'Feature Importance KNN: {knn}')# adjust the size of the plot (optional)
ax = sns.barplot(x='index', y=0, data=top_df_T)
labels = ax.get_xticklabels()
ax.set_xticklabels(labels, rotation=45, ha='right', rotation_mode='anchor',fontsize=14)
plt.xlabel('Category', fontsize=14)  # set x-axis label font size
plt.ylabel('Saliency', fontsize=14)
plt.tight_layout()
plt.show()

def output_results(results):
    avg_results_dict = {}
    avg_acc = 0.0
    avg_bacc = 0.0
    avg_f1 = 0.0
    avg_confusion_matrix = np.zeros((2, 2))
    avg_auc = 0.0
    subj_acc = 0.0
    subj_macro_acc = 0.0
    avg_acc_high = 0.0
    avg_acc_low = 0.0
    all_predictions = []
    all_labels = []
    all_IDs = []

    for key in results.keys():
        subj_macro_acc += results[key]['subject_macro_accuracy']
        subj_acc += results[key]['subject_accuracy']
        avg_acc += results[key]['accuracy']
        avg_bacc += results[key]['balanced_accuracy']
        avg_f1 += results[key]['f1-score']
        avg_confusion_matrix += results[key]['confusion_matrix']
        avg_acc_high += results[key]['acc>0.5']
        avg_acc_low += results[key]['acc<0.5']
        avg_auc += results[key]['auc']
        all_predictions.append(results[key]['predictions'])
        all_labels.append(results[key]['labels'])
        all_IDs.append(results[key]['subject_id'])

    avg_results_dict['subject_accuracy'] = subj_acc / len(folds.keys())
    avg_results_dict['accuracy'] = avg_acc / len(folds.keys())
    avg_results_dict['macro_accuracy'] = avg_bacc / len(folds.keys())
    avg_results_dict['f1-score'] = avg_f1 / len(folds.keys())
    avg_results_dict['confusion_matrix'] = avg_confusion_matrix / len(folds.keys())
    avg_results_dict['auc'] = avg_auc / len(folds.keys())
    avg_results_dict['subject_macro_accuracy'] = subj_macro_acc / len(folds.keys())
    avg_results_dict['high_acc'] = avg_acc_high / len(folds.keys())
    avg_results_dict['low_acc'] = avg_acc_low / len(folds.keys())

    acc_0 = [results[key]['accuracy'] for key in results.keys()]
    acc_1 = [results[key]['balanced_accuracy'] for key in results.keys()]
    acc_2 = [results[key]['acc>0.5'] for key in results.keys()]
    acc_3 = [results[key]['acc<0.5'] for key in results.keys()]
    f1 = [results[key]['f1-score'] for key in results.keys()]
    auc = [results[key]['auc'] for key in results.keys()]
    # STD balanced accuracy
    # STD acc > 0.5
    # STD acc < 0.5

    print(f'Accuracy: {torch.mean(torch.tensor(acc_0)).item():.3f} +- {torch.std(torch.tensor(acc_0)).item():.3f}')
    print(f'Balanced Accuracy: {torch.mean(torch.tensor(acc_1)).item():.3f} +- {torch.std(torch.tensor(acc_1)).item():.3f}')
    print(f'F1-score: {torch.mean(torch.tensor(f1)).item():.3f} +- {torch.std(torch.tensor(f1)).item():.3f}')
    print(f'AUC: {torch.mean(torch.tensor(auc)).item():.3f} +- {torch.std(torch.tensor(auc)).item():.3f}')
    print(f'Acc > {centering}: {torch.mean(torch.tensor(acc_2)).item():.3f} +- {torch.std(torch.tensor(acc_2)).item():.3f}')
    print(f'Acc < {centering}: {torch.mean(torch.tensor(acc_3)).item():.3f} +- {torch.std(torch.tensor(acc_3)).item():.3f}')

    all_labels = [item for sublist in all_labels for item in sublist]
    all_predictions = [item for sublist in all_predictions for item in sublist]
    all_IDs = [item for sublist in all_IDs for item in sublist]

    data = {'subject': all_IDs, 'Labels': all_labels, 'Prediction': all_predictions}

    # Create DataFrame
    df = pd.DataFrame(data)

    merged_preds_weights = pd.merge(output, df, on='subject')

    merged_preds_weights.to_csv(f'./analysis_lab_data/weights_pred_knn_{knn}_eigenbasis_{k}_centering_{centering}_lab_data.csv')
    return avg_results_dict
# Average the results over the 5 folds and print the metrics
print(f'Average results for {construct} and Knn {knn}:')
print(output_results(results))

print('Results per fold:')

print('Accuracy')
for key in results.keys():
    print(results[key]['accuracy'])

print('Balanced Accuracy')
for key in results.keys():
    print(results[key]['balanced_accuracy'])

print('F1 Score')
for key in results.keys():
    print(results[key]['f1-score'])

print('AUC')
for key in results.keys():
    print(results[key]['auc'])

print(f'ACC>{centering}')
for key in results.keys():
    print(results[key]['acc>0.5'])

print(f'ACC<{centering}')
for key in results.keys():
    print(results[key]['acc<0.5'])



