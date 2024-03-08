from load_data import *
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from datasets import *
from models import *
from train_test_utils import *
from tabulate import tabulate
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

run_with_seed = 1964 #random.randint(0, 4294967295)
torch.manual_seed(run_with_seed)
np.random.seed(run_with_seed)
torch.cuda.manual_seed(run_with_seed)

# Load data
path = './forecast_data/processed_with_imaging_fa_merged.csv'
additional_path = './forecast_data/additional_with_imaging_fa_merged.csv'

# Path to save the processed version of the dataset, if needed
w_path = ''

input_feats = load_longitudinal_tabular_data(input_path=path, write_path=w_path, quick=True, write_csv=False)
additional = load_longitudinal_tabular_data(input_path=additional_path, write_path=w_path, quick=True, write_csv=False)

# Because this variable should not be used
input_feats = input_feats.drop(columns=['aces_total'])

# We are predicting age and construct in a multi-task setting
age = False
plot_flag = True
# seq2seq dictates if we predict all time-steps or only the last one
seq2seq = False
batch_size = 25
additional_val = False

# age_range = [12, 18]
# sex = 2
# (input_feats['visit_age'] <= age_range[1]) & (input_feats['visit_age'] <= age_range[0])
# Options are 'negative_valence' and 'positive_valence'
# construct = 'positive_valence'
construct = 'cahalan_score_y'

# Load in additional validation
if additional_val:
    scaler = MinMaxScaler(feature_range=(-1,1))
    additional = additional.drop(columns=['aces_total'])
    partition_additional = list(np.unique(additional.loc[:, 'subject']))
    additional['cahalan_score_x'] = additional['cahalan_score_x'].replace(1, 0)
    X_additional = additional.iloc[:, 3:]
    X_additional = X_additional.drop(columns='cahalan_score_y')
    X_additional = scaler.fit_transform(X_additional)
    X_additional = pd.DataFrame(data=X_additional, columns=additional.columns[3:-1])
    X_additional = X_additional.set_index(additional.index)
    X_additional.insert(0, 'subject', additional.loc[:, 'subject'], True)
    labels_additional = {}
    subject_additional = {}
    params = {'shuffle': True,
            'num_workers': 0,
            'batch_size': 1}
    for key in partition_additional:
        subj_v_add = additional[additional['subject'] == key]
        subj_visits_add = X_additional[X_additional['subject'] == key]    
        labels_additional[key] = subj_v_add.loc[:, construct]
        subject_additional[key] = subj_visits_add.iloc[:, 1:]
    additional_validation_set = Dataset(partition_additional, subject_additional, labels_additional, {}, age)
    additional_validation_generator = torch.utils.data.DataLoader(additional_validation_set, **params)


labels_train = {}
labels_test = {}
labels_control_diseased = {}
ages = {}
# print(input_feats.loc[(input_feats['visit_age'] <= age_range[1]) & (input_feats['visit_age'] > age_range[0])].shape)
# print(input_feats.loc[(input_feats['visit_age'] <= age_range[1]) & (input_feats['visit_age'] > age_range[0]) & (input_feats['cahalan_score'] == 1)].shape)
for key in list(np.unique(input_feats.loc[:, 'subject'])):
    subj_v = input_feats[input_feats['subject'] == key]
    labels_train[key] = subj_v.loc[:, construct]
    # subj_v = input_feats.loc[(input_feats['visit_age'] <= age_range[1]) & (input_feats['visit_age'] > age_range[0]) & (input_feats['subject'] == key)]
    labels_test[key] = subj_v.loc[:, construct]
    ages[key] = subj_v.loc[:, 'visit_age']
    
split_stratified_labels = []
for key in list(np.unique(input_feats.loc[:, 'subject'])):
    split_stratified_labels.append(labels_train[key].max())

# Since if we are predicting the age as a task we don't use it as an input feature
if age:
    input_feats = input_feats.drop(columns=['visit_age'])
# We are doing 5-fold cross validation se we are creating dictionaries to save the data for each fold
partition = {}
folds = {}
subject_folds = {}
mfb_folds = {}
folds_of_the_labels_train = {}
folds_of_the_labels_test = {}
folds_of_the_ages = {}
counter = 0

# We are creating the object scaler to normalize our input data from -1 to 1
scaler = MinMaxScaler(feature_range=(-1,1))

# We are performing stratified CV since we hace class imbalance
kf = StratifiedKFold(n_splits=5, shuffle=False)  # False for reproducible folds
X = np.array(list(np.unique(input_feats.loc[:, 'subject'])))
y = np.array(split_stratified_labels)
kf.get_n_splits(X, y)

# We want to separate the data to train and test sets for each fold, normalize them and also 
# exclude the 'fake control' subjects from the test set. 
# 'Fake controls' are subjects that have 'True' in any depression construct in at least one of the visits. 
# The four constructs are: 'positive_valence', 'negative_valence', 'arousal', 'cognitive' 
for train_index, test_index in kf.split(X, y):
    labels_train_fold = labels_train.copy()
    labels_test_fold = labels_test.copy()
    ages_fold = ages.copy()
    subject_ages = {}
    subject_post = {}
    train_subj = input_feats.loc[input_feats['subject'].isin(list(X[train_index]))]
    y_train = train_subj.loc[:, construct]
    test_subj = input_feats.loc[input_feats['subject'].isin(list(X[test_index]))]
    # test_subj = test_subj.loc[(input_feats['visit_age'] <= age_range[1]) & (input_feats['visit_age'] > age_range[0])]
    train_subj = train_subj.drop(columns='cahalan_score_y')
    test_subj = test_subj.drop(columns='cahalan_score_y')

    # In this version of the dataset the 21 first columns contain features that should not be used as input 
    # for the prediction.
    # These features are the personal information, like sex, race etc., the labels that we are predicting and other
    # potential prediction variables.
    X_train = train_subj.iloc[:, 3:]
    X_train = scaler.fit_transform(X_train)
    
    X_train = pd.DataFrame(data=X_train, columns=train_subj.columns[3:])
    X_train = X_train.set_index(train_subj.index)
    X_train.insert(0, 'subject', train_subj.loc[:, 'subject'], True)

    X_test = test_subj.iloc[:, 3:]
    X_test = scaler.transform(X_test)
    
    X_test = pd.DataFrame(data=X_test, columns=test_subj.columns[3:])
    X_test = X_test.set_index(test_subj.index)
    X_test.insert(0, 'subject', test_subj.loc[:, 'subject'], True)

    partition['test'] = list()
    partition['train'] = list()
    for subject in input_feats.subject.unique():
        if subject in list(X[train_index]):
            subj_visits = X_train[X_train['subject'] == subject]
            subject_ages[subject] = subj_visits
            partition['train'].append(subject)
            
        elif subject in list(X[test_index]):
            subj_visits = X_test[X_test['subject'] == subject]
            subject_ages[subject] = subj_visits
            partition['test'].append(subject)

    folds[counter] = partition.copy()

    # Subject-specific dataset with all the visits and post-processed for training
    for key in list(partition['train'] + partition['test']):
        df = subject_ages[key]
        df = df.iloc[:, 1:]
        subject_post[key] = df

    subject_folds[counter] = subject_post.copy()
    folds_of_the_labels_train[counter] = labels_train_fold.copy()
    folds_of_the_labels_test[counter] = labels_test_fold.copy()
    folds_of_the_ages[counter] = ages_fold.copy()

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
df_list = []
results_additional = {}

# Train models for each fold

for fold in folds.keys():

    # Get the data per fold
    partition = folds[fold]
    subject_post = subject_folds[fold]
    pos_weight = mfb_folds[fold]
    labels_train_f = folds_of_the_labels_train[fold]
    labels_test_f = folds_of_the_labels_test[fold]
    ages_f = folds_of_the_ages[fold]

    # Dataset generators
    training_set = Dataset(partition['train'], subject_post, labels_train_f, ages_f, age)
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    validation_set = Dataset(partition['test'], subject_post, labels_test_f, ages_f, age)
    validation_generator = torch.utils.data.DataLoader(validation_set, **params)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # The parameters I trained with, after hyperparameter tuning
    epoch = 0
    max_epochs = 100

    feature_dim = 128
    input_dim = next(iter(training_generator))[0].shape[2]
    output_dim = 1
    n_layers = 1
    hidden_dim = 64

    
    if age:
        model = AgeGRUNet(feature_dim=feature_dim, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                          n_layers=n_layers, seq2seq=seq2seq, device=device, drop_prob=0.0)
    else:
        model = GRUNet(feature_dim=feature_dim, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                          n_layers=n_layers, seq2seq=seq2seq, device=device, drop_prob=0)

    # Loss chosen for binary classification task
    # score_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).float().to(device))
    score_criterion = nn.BCEWithLogitsLoss()

    if age:
        # Loss chosen for age prediction
        age_criterion = nn.MSELoss()
        criterion = {}
        criterion['score'] = score_criterion
        criterion['age'] = age_criterion
    else:
        criterion = score_criterion

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)
    model.to(device)

    model.train()
    
    if age:
        model_trained, h, plot = train_gru_age(model=model, criterion=criterion, optimizer=optimizer, max_epochs=max_epochs,
                                         train_loader=training_generator, val_loader=validation_generator,
                                         device=device, seq2seq=seq2seq, params=params, batch_size=batch_size)
    else:
        model_trained, h, plot = train_gru(model=model, criterion=criterion, optimizer=optimizer, max_epochs=max_epochs,
                                         train_loader=training_generator, val_loader=validation_generator,
                                         device=device, seq2seq=seq2seq, params=params, batch_size=batch_size, scheduler=scheduler)
    if plot_flag:               
        fig, ax = plt.subplots()
        ax.plot(plot['epoch'], plot['train_loss'], label='train')
        ax.plot(plot['epoch'], plot['val_loss'], label='validation')
        ax.set(xlabel='epoch', ylabel='loss',
            title='traning curve fold {}'.format(fold))
        ax.legend()
        fig.savefig("./plots/train_curve{}.png".format(fold)) 
    
    # Path to save the models
    if age:
        output_path = f'{construct}_fold_{fold}_tabular_no_aces.ckpt'
    else:
        output_path = f'{construct}_fold_{fold}_tabular_no_aces_no_age.ckpt'
    
    print(f'Model training complete for fold {fold} complete.')
    # Uncomment to save models
    #torch.save(model_trained.state_dict(), output_path)
    #print(f'Model Saved at: {output_path}')
    
    if seq2seq:
        if age:
            results[f'split{fold}'] = evaluate_all_timesteps_age_per_subject(model=model_trained, val_loader=validation_generator,
                                                                 hidden=h, device=device)
        else:
            results[f'split{fold}'] = evaluate_all_timesteps_per_subject(model=model_trained, val_loader=validation_generator, hidden=h, device=device)
    else:
        results[f'split{fold}'] = evaluate_last_timestep(model=model_trained, val_loader=validation_generator, device=device, criterion=criterion)
        if additional_val:
            results_additional[f'split{fold}'] = evaluate_last_timestep(model=model_trained, val_loader=additional_validation_generator, device=device, criterion=criterion)
    df = pd.DataFrame.from_dict({k:v for k,v in results[f'split{fold}'].items() if k in ['subject', 'probabilities', 'true_values']})
    df_list.append(df)

def output_results(results):
    avg_results_dict = {}
    avg_acc = 0.0
    avg_bacc = 0.0
    avg_f1 = 0.0
    avg_confusion_matrix = np.zeros((2, 2))
    subj_acc = 0.0
    subj_macro_acc = 0.0
    avg_auc = 0.0

    for key in results.keys():
        subj_macro_acc += results[key]['subject_macro_accuracy']
        subj_acc += results[key]['subject_accuracy']
        avg_acc += results[key]['accuracy']
        avg_bacc += results[key]['balanced_accuracy']
        avg_f1 += results[key]['f1-score']
        avg_confusion_matrix += results[key]['confusion_matrix']
        avg_auc +=  results[key]['auc']

    avg_results_dict['subject_accuracy'] = subj_acc / len(folds.keys())
    avg_results_dict['subject_macro_accuracy'] = subj_macro_acc / len(folds.keys())
    avg_results_dict['accuracy'] = avg_acc / len(folds.keys())
    avg_results_dict['macro_accuracy'] = avg_bacc / len(folds.keys())
    avg_results_dict['f1-score'] = avg_f1 / len(folds.keys())
    avg_results_dict['confusion_matrix'] = avg_confusion_matrix / len(folds.keys())
    avg_results_dict['auc'] = avg_auc / len(folds.keys())
    acc_1 = [results[key]['balanced_accuracy'] for key in results.keys()]


    print(torch.std(torch.tensor(acc_1)).item())
    return avg_results_dict
# Average the results over the 5 folds and print the metrics

print(f'Average results for {construct}:')
print(output_results(results))
if additional_val:
    print(output_results(results_additional))

