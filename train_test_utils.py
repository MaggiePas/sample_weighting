import warnings
import torch
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, mean_squared_error, confusion_matrix
import itertools
import numpy as np
import higher
import shap
from sklearn import metrics
from captum.attr import IntegratedGradients, GuidedBackprop, DeepLift, FeatureAblation, Saliency
# from torchray.attribution.guided_backprop import GuidedBackpropContext
from torch import nn

def binary_acc(y_pred, y_test, seq2seq):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    if seq2seq:
        correct_results_sum = (y_pred_tag == y_test._base).sum().float()
        acc = correct_results_sum / len(y_test)
    else:
        correct_results_sum = (y_pred_tag == y_test).sum().float()
        acc = correct_results_sum / y_test.unsqueeze(axis=0).shape[0]
    acc = torch.round(acc * 100)

    return acc


# https://stackoverflow.com/questions/58172188/how-to-add-l1-regularization-to-pytorch-nn-model
def l1_regularizer(model, lambda_l1=0.01, weight_or_bias='weight'):
    lossl1 = 0
    for model_param_name, model_param_value in model.named_parameters():
            if weight_or_bias in model_param_name:
                lossl1 += lambda_l1 * model_param_value.abs().sum()
    return lossl1

def train_gru_age(model=None, criterion=None, optimizer=None, max_epochs=30, train_loader=None, val_loader=None, device=None,
                  params=None):

    # Loop over epochs
    for epoch in range(max_epochs):
        # Training
        avg_loss = 0.
        avg_acc = 0.
        counter = 0

        for local_batch, local_labels, local_ages in train_loader:
            counter += 1
            h = model.init_hidden(batch_size=local_batch.shape[0])
            h = h.to(device)
            local_batch, local_labels, local_ages = local_batch.to(device), local_labels.to(device), local_ages.to(device)
            local_batch = local_batch.squeeze(axis=0)
            local_labels = local_labels.squeeze(axis=0)
            local_batch = local_batch.unsqueeze(axis=1)
            local_labels = local_labels.unsqueeze(axis=1)
            local_ages = local_ages.squeeze(axis=0)

            optimizer.zero_grad()

            h = h.data
            out, h, out_age = model(local_batch.float(), h.float())

            loss_score = criterion['score'](out.squeeze().to(device), local_labels.squeeze().float().to(device))
            loss_age = criterion['age'](out_age.squeeze(), local_ages.squeeze().float())

            # The loss is a combination of the BCE and the MSE for the age.
            # I have weighted higher the BCE loss in this case after hyperparameter tuning
            loss = loss_score + 0.2*loss_age

            acc = binary_acc(out.squeeze(), local_labels)

            # I am adding L1 regularization for the weights to minimize overfitting since out dataset is so small
            total_loss = loss + l1_regularizer(model, lambda_l1=0.001, weight_or_bias='weight')

            total_loss.backward()
            optimizer.step()
            avg_loss += total_loss.item()
            avg_acc += acc.item()

            # Select how often you want your results to be printed during training
            if counter % 6000 == 0:
                print(
                    "Epoch {}... Step: {}/{}... Average Loss for Epoch: {}... Accuracy: {}".format(epoch, counter, len(train_loader),
                                                                                         avg_loss / counter, avg_acc / counter))
                #evaluate_all_timesteps_age(model=model, val_loader=val_loader, hidden=h, device=device)
                # This function returns also the subject-level accuracy and macro accuracy
                evaluate_all_timesteps_age_per_subject(model=model, val_loader=val_loader, hidden=h, device=device)

                model.train()
    return model, h


def evaluate_all_timesteps_age_per_subject(model=None, val_loader=None, hidden=None, device=None):
    y_pred_list = []
    y_pred_ages = []
    y_test = []
    y_test_ages = []
    model.eval()
    subject_acc = []
    subject_bacc_control = []
    subject_bacc_diseased = []
    with torch.no_grad():
        for local_batch, local_labels, local_ages in val_loader:
            h = model.init_hidden(batch_size=local_batch.shape[0])
            h = h.to(device)
            local_batch, local_labels, local_ages = local_batch.to(device), local_labels.to(device), local_ages.to(device)
            local_batch = local_batch.squeeze(axis=0)
            local_labels = local_labels.squeeze(axis=0)
            local_batch = local_batch.unsqueeze(axis=1)
            local_labels = local_labels.unsqueeze(axis=1)
            local_ages = local_ages.squeeze(axis=0)

            h = h.data
            out, h, out_ages = model(local_batch.float(), h.float())
            y_pred_tag = torch.round(torch.sigmoid(out.squeeze()))
            out_ages = out_ages.squeeze()

            # We need to change the labels from an array of arrays to a normal array otherwise the accuracy is not
            # calculated correctly
            label_list = local_labels._base.cpu().numpy().tolist()
            label_list = list(itertools.chain.from_iterable(label_list))

            # Create a list of all the predictions and calculate the subject-level accuracy over all visits
            if len(y_pred_tag.shape) == 0:
                y_pred_list = y_pred_list + [(y_pred_tag.cpu().numpy().tolist())]
                y_pred_ages = y_pred_ages + [(out_ages.cpu().numpy().tolist())]
                subject_acc.append(accuracy_score(label_list, [y_pred_tag.cpu().numpy()]))
            else:
                y_pred_list = y_pred_list + (y_pred_tag.cpu().numpy().tolist())
                y_pred_ages = y_pred_ages + (out_ages.cpu().numpy().tolist())
                subject_acc.append(accuracy_score(label_list, y_pred_tag.cpu().numpy()))

            # Keep the accuracies of control and diseased subjects separately so we can calculate the
            # overall subject-level macro accuracy
            if torch.sum(local_labels) > 0:
                subject_bacc_diseased.append(subject_acc[-1])
            else:
                subject_bacc_control.append(subject_acc[-1])

            y_test = y_test + (local_labels._base.cpu().numpy().tolist())
            y_test_ages = y_test_ages + (local_ages._base.cpu().numpy().tolist())

    y_test = list(itertools.chain.from_iterable(y_test))
    y_test_ages = list(itertools.chain.from_iterable(y_test_ages))
    subject_macro_accuracy = (sum(subject_bacc_diseased)/len(subject_bacc_diseased)
                              + sum(subject_bacc_control)/len(subject_bacc_control))/2
    results_dict = {'subject_accuracy': sum(subject_acc)/len(subject_acc),
                    'subject_macro_accuracy': subject_macro_accuracy,
                    'accuracy': accuracy_score(y_test, y_pred_list),
                    'balanced_accuracy': balanced_accuracy_score(y_test, y_pred_list),
                    'f1-score': f1_score(y_test, y_pred_list, average='macro'),
                    'mse-age': mean_squared_error(np.array(y_test_ages), np.array(y_pred_ages))}

    # Subject-level means we calculate one score per subject over all visits.
    # That way, subjects with more visits do not influence the results more than subjects with just one visit
    # Afterwards we also calculate the overall results over all visits, regardless of subject

    return results_dict


def evaluate_all_timesteps_per_subject(model=None, val_loader=None, hidden=None, device=None):
    y_pred_list = []
    y_test = []
    model.eval()
    subject_acc = []
    subject_bacc_control = []
    subject_bacc_diseased = []
    with torch.no_grad():
        for local_batch, local_labels in val_loader:
            h = model.init_hidden(batch_size=local_batch.shape[0])
            h = h.to(device)
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            # if local_batch.shape[1] == 0:
            #     continue

            local_batch = local_batch.squeeze(axis=0)
            local_labels = local_labels.squeeze(axis=0)
            local_batch = local_batch.unsqueeze(axis=1)
            local_labels = local_labels.unsqueeze(axis=1)

            h = h.data
            out, h = model(local_batch.float(), h.float())
            y_pred_tag = torch.round(torch.sigmoid(out.squeeze()))

            # We need to change it for an array of arrays to a normal array otherwise the accuracy is not
            # calculated correctly
            patata = local_labels._base.cpu().numpy().tolist()
            patata = list(itertools.chain.from_iterable(patata))

            # Create a list of all the predictions and calculate the subject-level accuracy
            if len(y_pred_tag.shape) == 0:
                y_pred_list = y_pred_list + [(y_pred_tag.cpu().numpy().tolist())]
                subject_acc.append(accuracy_score(patata, [y_pred_tag.cpu().numpy()]))
            else:
                y_pred_list = y_pred_list + (y_pred_tag.cpu().numpy().tolist())
                subject_acc.append(accuracy_score(patata, y_pred_tag.cpu().numpy()))

            # Keep the accuracies of control and diseased subjects separately so we can calculate the
            # overall subject-level macro accuracy
            if torch.sum(local_labels) > 0:
                subject_bacc_diseased.append(subject_acc[-1])
            else:
                subject_bacc_control.append(subject_acc[-1])

            y_test = y_test + (local_labels._base.cpu().numpy().tolist())
            
    y_test = list(itertools.chain.from_iterable(y_test))
    if len(subject_bacc_diseased) == 0:
        subject_macro_accuracy = sum(subject_bacc_control)/len(subject_bacc_control)
    else:
        subject_macro_accuracy = (sum(subject_bacc_diseased)/len(subject_bacc_diseased)
                                + sum(subject_bacc_control)/len(subject_bacc_control))/2
    results_dict = {'subject_accuracy': sum(subject_acc)/len(subject_acc),
                    'subject_macro_accuracy': subject_macro_accuracy,
                    'accuracy': accuracy_score(y_test, y_pred_list),
                    'balanced_accuracy': balanced_accuracy_score(y_test, y_pred_list),
                    'f1-score': f1_score(y_test, y_pred_list, average='macro')}
    
    return results_dict


def acc_per_run_dataset_cross_sectional():

    baseline_acc = 0.761871
    baseline_bacc = 0.722522
    baseline_f1 = 0.64044

    return baseline_acc, baseline_bacc, baseline_f1


def evaluate_last_timestep(model=None, val_loader=None, device=None, criterion=None, lookup=None, weight=None, basis=None, explainer=None, centering=None):
    subj_id_list = []
    y_pred_list = []
    y_pred_auc = []
    y_pred_list_ge = []
    y_pred_list_le = []
    y_test = []
    y_test_ge = []
    y_test_le = []
    error_set = []
    model.eval()
    with torch.no_grad():
        # No age

        if len(next(iter(val_loader))) == 2 or len(next(iter(val_loader))) == 3:
            for local_batch, local_labels, ID in val_loader:
                local_batch = local_batch.to(device)
                # local_labels = torch.max(local_labels)
                local_labels = local_labels[:, -1].to(device)

                h = model.init_hidden(batch_size=local_batch.shape[0])
                h = h.to(device)
                out, h = model(local_batch.float(), h.float())
                y_pred_tag = torch.round(torch.sigmoid(out.squeeze()))
                if y_pred_tag != local_labels:
                    error_set.append(ID[0])
                if basis is not None:
                    q1 = torch.quantile(centering + weight @ basis.T, 0.25, interpolation='nearest').item()
                    q4 = torch.quantile(centering + weight @ basis.T, 0.75, interpolation='nearest').item()
                    median = torch.median(centering + weight @ basis.T).item()
                    input_weight = centering + weight @ basis[lookup[ID[0]]]
                    if input_weight >= median:
                        y_pred_list_ge.append(y_pred_tag.cpu().numpy())
                        y_test_ge.append(local_labels.cpu().numpy())
                    elif input_weight <= median:
                        y_pred_list_le.append(y_pred_tag.cpu().numpy())
                        y_test_le.append(local_labels.cpu().numpy())
                y_pred_list.append(y_pred_tag.cpu().numpy())
                subj_id_list.append(ID)
                y_pred_auc.append(torch.sigmoid(out.squeeze()).cpu().numpy())
                y_test.append(local_labels.cpu().numpy())

    subj_id_list = [a[0] for a in subj_id_list]
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    y_test = [a.squeeze().tolist() for a in y_test]
    y_pred_list_ge = [a.squeeze().tolist() for a in y_pred_list_ge]
    y_test_ge = [a.squeeze().tolist() for a in y_test_ge]
    y_pred_list_le = [a.squeeze().tolist() for a in y_pred_list_le]
    y_test_le = [a.squeeze().tolist() for a in y_test_le]
    loss = criterion(out.squeeze().to(device), local_labels.squeeze().float().to(device))
    # In this case subject accuracy and accuracy are the same since we have only one visit per subject
    if basis is not None:
        results_dict = {'subject_accuracy': accuracy_score(y_test, y_pred_list),
                        'subject_macro_accuracy': balanced_accuracy_score(y_test, y_pred_list),
                        'accuracy': accuracy_score(y_test, y_pred_list),
                        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred_list),
                        'f1-score': f1_score(y_test, y_pred_list, average='macro'),
                        'confusion_matrix': confusion_matrix(y_test, y_pred_list),
                        'auc': metrics.roc_auc_score(y_test, y_pred_auc),
                        'loss': (loss + l1_regularizer(model, lambda_l1=0.001, weight_or_bias='weight')).detach().numpy(),
                        'acc>0.5': accuracy_score(y_test_ge, y_pred_list_ge),
                        'acc<0.5': accuracy_score(y_test_le, y_pred_list_le),
                        'bacc>0.5': balanced_accuracy_score(y_test_ge, y_pred_list_ge),
                        'bacc<0.5': balanced_accuracy_score(y_test_le, y_pred_list_le),
                        'predictions': y_pred_list,
                        'labels': y_test,
                        'subject_id': subj_id_list}
    else:
        results_dict = {'subject_accuracy': accuracy_score(y_test, y_pred_list),
                'subject_macro_accuracy': balanced_accuracy_score(y_test, y_pred_list),
                'accuracy': accuracy_score(y_test, y_pred_list),
                'balanced_accuracy': balanced_accuracy_score(y_test, y_pred_list),
                'f1-score': f1_score(y_test, y_pred_list, average='macro'),
                'confusion_matrix': confusion_matrix(y_test, y_pred_list),
                'auc': metrics.roc_auc_score(y_test, y_pred_auc),
                'loss': (loss + l1_regularizer(model, lambda_l1=0.001, weight_or_bias='weight')).detach().numpy(),
                'predictions': y_pred_list,
                'labels': y_test,
                'error_set': error_set
                }
    #print(results_dict)
    return results_dict


def evaluate_last_timestep_shap(model=None, val_loader=None, device=None, criterion=None, lookup=None, weight=None, basis=None, explainer=None, centering=None):
    y_pred_list = []
    y_pred_auc = []
    y_pred_list_ge = []
    y_pred_list_le = []
    y_test = []
    y_test_ge = []
    y_test_le = []
    model.eval()

    with torch.no_grad():
        # No age
        shap_values_fold = []
        if len(next(iter(val_loader))) == 2 or len(next(iter(val_loader))) == 3:
            for local_batch, local_labels, ID in val_loader:
                local_batch = local_batch.to(device)
                # local_labels = torch.max(local_labels)
                local_labels = local_labels[:, -1].to(device)

                h = model.init_hidden(batch_size=local_batch.shape[0])
                h = h.to(device)
                out, h = model(local_batch.float(), h.float())
                y_pred_tag = torch.round(torch.sigmoid(out.squeeze()))
                if basis is not None:
                    q1 = torch.quantile(centering + weight @ basis.T, 0.25, interpolation='nearest').item()
                    q4 = torch.quantile(centering + weight @ basis.T, 0.75, interpolation='nearest').item()
                    median = torch.median(centering + weight @ basis.T).item()
                    input_weight = centering + weight @ basis[lookup[ID[0]]]
                    if input_weight >= median:
                        y_pred_list_ge.append(y_pred_tag.cpu().numpy())
                        y_test_ge.append(local_labels.cpu().numpy())
                        # if local_labels == y_pred_tag:
                        #     grad_backprop_attributions = explainer.attribute(local_batch.float())
                        #     shap_values_fold.append(abs(grad_backprop_attributions[0, -1, :]))

                    elif input_weight <= median:
                        y_pred_list_le.append(y_pred_tag.cpu().numpy())
                        y_test_le.append(local_labels.cpu().numpy())
                y_pred_list.append(y_pred_tag.cpu().numpy())
                y_pred_auc.append(torch.sigmoid(out.squeeze()).cpu().numpy())
                y_test.append(local_labels.cpu().numpy())

                if local_labels == y_pred_tag:
                    # for guided backprop + saliency + feature ablation
                    grad_backprop_attributions = explainer.attribute(local_batch.float())

                    # grad_backprop_attributions = explainer.attribute(local_batch.float(), n_steps=1, return_convergence_delta=False)
                    # local_batch.requires_grad = True
                    # model.eval()
                    # model.zero_grad()

                    # for integrated gradients
                    # grad_backprop_attributions = explainer.attribute(local_batch.float(), baselines=local_batch.float() * 0, target=0, return_convergence_delta=False)


                    # Get the mean of gradients across visits
                    # mean_x = abs(grad_backprop_attributions).mean(dim=1, keepdim=True)
                    # shap_values_fold.append(mean_x.squeeze())

                    # Get the gradients of last visit
                    shap_values_fold.append(abs(grad_backprop_attributions[0,-1,:]))


                # attributions, delta = ig.attribute(local_batch.float(), target=0, return_convergence_delta=True,n_steps=1)

                # SHAP Values computation for this fold
                # with warnings.catch_warnings():
                    # warnings.filterwarnings("ignore")
                    # Compute the SHAP values over the test data of the fold X_scaled_test
                    # shap_values_deep =s explainer.shap_values(local_batch.numpy())
                    # print(np.shape(local_batch.numpy()[0,:,:]))
                    # shap_values_fold.append(shap_values_deep)

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    y_test = [a.squeeze().tolist() for a in y_test]
    y_pred_list_ge = [a.squeeze().tolist() for a in y_pred_list_ge]
    y_test_ge = [a.squeeze().tolist() for a in y_test_ge]
    y_pred_list_le = [a.squeeze().tolist() for a in y_pred_list_le]
    y_test_le = [a.squeeze().tolist() for a in y_test_le]
    loss = criterion(out.squeeze().to(device), local_labels.squeeze().float().to(device))
    # In this case subject accuracy and accuracy are the same since we have only one visit per subject
    if basis is not None:
        results_dict = {'shap_fold': shap_values_fold,
            'subject_accuracy': accuracy_score(y_test, y_pred_list),
                        'subject_macro_accuracy': balanced_accuracy_score(y_test, y_pred_list),
                        'accuracy': accuracy_score(y_test, y_pred_list),
                        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred_list),
                        'f1-score': f1_score(y_test, y_pred_list, average='macro'),
                        'confusion_matrix': confusion_matrix(y_test, y_pred_list),
                        'auc': metrics.roc_auc_score(y_test, y_pred_auc),
                        'loss': (loss + l1_regularizer(model, lambda_l1=0.001, weight_or_bias='weight')).detach().numpy(),
                        'acc>0.5': accuracy_score(y_test_ge, y_pred_list_ge),
                        'acc<0.5': accuracy_score(y_test_le, y_pred_list_le)}
    else:
        results_dict = {'shap_fold': shap_values_fold,
            'subject_accuracy': accuracy_score(y_test, y_pred_list),
                'subject_macro_accuracy': balanced_accuracy_score(y_test, y_pred_list),
                'accuracy': accuracy_score(y_test, y_pred_list),
                'balanced_accuracy': balanced_accuracy_score(y_test, y_pred_list),
                'f1-score': f1_score(y_test, y_pred_list, average='macro'),
                'confusion_matrix': confusion_matrix(y_test, y_pred_list),
                'auc': metrics.roc_auc_score(y_test, y_pred_auc),
                'loss': (loss + l1_regularizer(model, lambda_l1=0.001, weight_or_bias='weight')).detach().numpy()}
    #print(results_dict)
    return results_dict

# Meta-weighting baseline
def train_gru_mod(model=None, criterion=None, optimizer=None, max_epochs=30, train_loader=None, val_loader=None, device=None, seq2seq=True, params=None, batch_size=None, scheduler=None \
                  , meta_loader=None):
    # Loop over epochs
    plot_epoch = []
    plot_train_acc = []
    plot_val_acc = []
    plot_val_loss = []
    plot_loss = []
    for epoch in range(max_epochs):
        # Training
        avg_loss = 0.
        another_avg_loss = 0.
        avg_acc = 0.
        avg_acc_ge = 0.
        avg_acc_le = 0.
        counter = 0
        counter_ge = 0
        counter_le = 0
        total_loss = 0

        w = torch.zeros(len(train_loader))
        start_idx = 0
        end_idx = start_idx + batch_size
        end = False
        weight_loopup = {}
        while True:
            if end_idx == len(train_loader):
                end = True
            with higher.innerloop_ctx(model, optimizer) as (meta_model, meta_opt):
                meta_loss = 0.0
                eps = torch.zeros(end_idx - start_idx, requires_grad=True)
                for i, (local_batch, local_labels, ID) in enumerate(train_loader):
                    weight_loopup[ID[0]] = i
                    if i >= start_idx and i < end_idx:
                        h = model.init_hidden(batch_size=params['batch_size'])
                        h = h.to(device)
                        local_labels = local_labels[:, -1].to(device)
                        meta_output, h = meta_model(local_batch.float(), h.float())
                        meta_loss += eps[i % batch_size] * criterion(meta_output.squeeze().to(device), local_labels.squeeze().float().to(device))
                    elif i >= end_idx:
                        break
                meta_opt.step(meta_loss)
                meta_loss_val = 0.0
                for i, (local_batch, local_labels, ID) in enumerate(meta_loader):
                    if i >= start_idx and i < end_idx:
                        h = model.init_hidden(batch_size=params['batch_size'])
                        h = h.to(device)
                        local_labels = local_labels[:, -1].to(device)
                        meta_output, h = meta_model(local_batch.float(), h.float())
                        meta_loss_val += criterion(meta_output.squeeze().to(device), local_labels.squeeze().float().to(device))
                    elif i >= end_idx:
                        break
                eps_grads = torch.autograd.grad(meta_loss_val, eps)[0].detach()         
                
                w_tilde = torch.clamp(-eps_grads, min=0)
                ll_norm = torch.sum(w_tilde)
                if ll_norm != 0:
                    w[start_idx:end_idx] = w_tilde / ll_norm
                else:
                    w[start_idx:end_idx] = w_tilde

                start_idx += batch_size
                end_idx += batch_size
                if end_idx > len(train_loader):
                    end_idx = len(train_loader)
                if end:
                    break
        median = torch.median(w).item()
        for subject in weight_loopup:
            weight_loopup[subject] = w[weight_loopup[subject]].item()
        for local_batch, local_labels, ID in train_loader:

            counter += 1
            h = model.init_hidden(batch_size=params['batch_size'])
            h = h.to(device)
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            if seq2seq:
                local_batch = local_batch.squeeze(axis=0)
                local_labels = local_labels.squeeze(axis=0)
                local_batch = local_batch.unsqueeze(axis=1)
                local_labels = local_labels.unsqueeze(axis=1)
            else:
                local_labels = local_labels[:, -1].to(device)

            optimizer.zero_grad()
            h = h.data
            out, h = model(local_batch.float(), h.float())
            loss = w[counter-1] * criterion(out.squeeze().to(device), local_labels.squeeze().float().to(device))
            another_loss = criterion(out.squeeze().to(device), local_labels.squeeze().float().to(device))

            acc = 0.01 * binary_acc(out.squeeze(), local_labels, seq2seq)

            total_loss += loss + 0.01 * l1_regularizer(model, lambda_l1=0.001, weight_or_bias='weight')
            if counter % batch_size == 0 or counter == len(train_loader):
                total_loss.backward()
                optimizer.step()
                total_loss = 0
            avg_loss += (loss + 0.01 * l1_regularizer(model, lambda_l1=0.001, weight_or_bias='weight')).item()
            another_avg_loss += (another_loss + l1_regularizer(model, lambda_l1=0.001, weight_or_bias='weight')).item()
            avg_acc += acc.item()
            if w[counter-1] > median:
                counter_ge += 1
                avg_acc_ge += acc.item()
            elif w[counter-1] <= median:
                counter_le += 1
                avg_acc_le += acc.item()
        result = evaluate_last_timestep(model=model, val_loader=val_loader, device=device, criterion=criterion)
        if epoch % 5 == 0 or epoch == max_epochs - 1:
            print(
                "Epoch {}... Average Loss for Epoch: {:.4f}... Train Accuracy: {:.4f}... Validation Accuracy: {}... Train Acc >0.5: {:.4f} | Train Acc <0.5: {:.4f} ".format(epoch, \
                    another_avg_loss / counter, avg_acc / counter, result['accuracy'], avg_acc_ge / counter_ge, avg_acc_le / counter_le))
        plot_loss.append((loss + l1_regularizer(model, lambda_l1=0.001, weight_or_bias='weight')).detach().numpy())
        plot_epoch.append(epoch)
        plot_train_acc.append(avg_acc / counter)
        plot_val_acc.append(result['accuracy'])
        plot_val_loss.append(result['loss'])
        scheduler.step()
    plot = {'epoch': plot_epoch, 'train_acc': plot_train_acc, 'val_acc': plot_val_acc, 'train_loss': plot_loss, 'val_loss': plot_val_loss}
    return model, h, plot, weight_loopup

def train_gru(model=None, criterion=None, optimizer=None, max_epochs=30, train_loader=None, val_loader=None, device=None, seq2seq=True, params=None, batch_size=None, scheduler=None, jtt=None):
    # Loop over epochs
    plot_epoch = []
    plot_train_acc = []
    plot_val_acc = []
    plot_val_loss = []
    plot_loss = []
    for epoch in range(max_epochs):
        # Training
        avg_loss = 0.
        avg_acc = 0.
        counter = 0
        batch_counter = 0
        total_loss = 0

        for local_batch, local_labels, _ in train_loader:
            counter += 1
            h = model.init_hidden(batch_size=params['batch_size'])
            h = h.to(device)
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            if seq2seq:
                local_batch = local_batch.squeeze(axis=0)
                local_labels = local_labels.squeeze(axis=0)
                local_batch = local_batch.unsqueeze(axis=1)
                local_labels = local_labels.unsqueeze(axis=1)
            else:
                local_labels = local_labels[:, -1].to(device)

            optimizer.zero_grad()
            h = h.data
            out, h = model(local_batch.float(), h.float())
            loss = criterion(out.squeeze().to(device), local_labels.squeeze().float().to(device))

            acc = 0.01 * binary_acc(out.squeeze(), local_labels, seq2seq)

            total_loss += loss + l1_regularizer(model, lambda_l1=0.001, weight_or_bias='weight')
            if counter % batch_size == 0 or counter == len(train_loader):
                total_loss.backward()
                optimizer.step()
                total_loss = 0
            avg_loss += (loss + l1_regularizer(model, lambda_l1=0.001, weight_or_bias='weight')).item()
            avg_acc += acc.item()

            if counter % 1000 == 0:
                print(
                    "Epoch {}... Step: {}/{}... Average Loss for Epoch: {}... Accuracy: {}".format(epoch, counter, len(train_loader),
                                                                                         avg_loss / counter, avg_acc / counter))
                if seq2seq:
                    evaluate_all_timesteps_per_subject(model=model, val_loader=val_loader, hidden=h, device=device)
                else:
                    evaluate_last_timestep(model=model, val_loader=val_loader, device=device)

                model.train()
        result = evaluate_last_timestep(model=model, val_loader=val_loader, device=device, criterion=criterion)
        if epoch % 5 == 0 or epoch == max_epochs - 1:
            print(
                "Epoch {}... Average Loss for Epoch: {:.4f}... Train Accuracy: {:.4f}... Validation Accuracy: {}...".format(epoch, \
                    avg_loss / counter, avg_acc / counter, result['accuracy']))
        plot_loss.append((loss + l1_regularizer(model, lambda_l1=0.001, weight_or_bias='weight')).detach().numpy())
        plot_epoch.append(epoch)
        plot_train_acc.append(avg_acc / counter)
        plot_val_acc.append(result['accuracy'])
        plot_val_loss.append(result['loss'])
        scheduler.step()
    plot = {'epoch': plot_epoch, 'train_acc': plot_train_acc, 'val_acc': plot_val_acc, 'train_loss': plot_loss, 'val_loss': plot_val_loss}

    if jtt is not None:
        result_train = evaluate_last_timestep(model=model, val_loader=train_loader, device=device, criterion=criterion)
        return model, h, plot, result_train['error_set']
    else:
        return model, h, plot


class ModelWrapper(nn.Module):
    def __init__(self, model, h):
        super().__init__()
        self.model = model

        # for integrated gradients
        # self.h = self.model.init_hidden(50)

        # for guided backprop and saliency and feature ablation
        self.h = self.model.init_hidden(1)
        self.h = self.h.detach()  # Detach the hidden state from its history

    def forward(self, x):
        # Make sure x requires gradients
        # x.requires_grad = True
        out, _ = self.model(x, self.h)
        return out

def train_gru_spectral(model=None, criterion=None, optimizer=None, max_epochs=30, train_loader=None, val_loader=None, device=None, seq2seq=True, params=None, batch_size=None, scheduler=None, \
    weight=None, lookup=None, lookup_val=None, basis=None, basis_val=None, eigenvalues=None, mode=1, negative_alpha=1, alpha=1, centering=0.5):
    # Loop over epochs
    plot_epoch = []
    plot_train_acc = []
    plot_val_acc = []
    plot_val_loss = []
    plot_loss = []
    SHAP_values_per_fold_deep = []

    for epoch in range(max_epochs):
        # Training
        avg_loss = 0.
        avg_acc = 0.
        avg_acc_ge = 0.
        avg_acc_le = 0.
        counter = 0
        counter_ge = 0
        counter_le = 0
        total_loss = 0

        for local_batch, local_labels, ID in train_loader:
            counter += 1
            h = model.init_hidden(batch_size=params['batch_size'])
            h = h.to(device)
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            if seq2seq:
                local_batch = local_batch.squeeze(axis=0)
                local_labels = local_labels.squeeze(axis=0)
                local_batch = local_batch.unsqueeze(axis=1)
                local_labels = local_labels.unsqueeze(axis=1)
            else:
                local_labels = local_labels[:, -1].to(device)

            optimizer.zero_grad()
            h = h.data
            out, h = model(local_batch.float(), h.float())
            # Create the model wrapper
            # ID is a tuple for some reason, and we lookup the basis row index for the particular subject
            # dot product the relevant basis to the a_is
            median = torch.median(centering + weight @ basis.T).item()
            input_weight = centering + weight @ basis[lookup[ID[0]]]
            negative_penalty = negative_alpha * torch.sum(torch.nn.functional.relu(-1 * input_weight))
            if mode == 1:
                # loss = criterion(out.squeeze().to(device), local_labels.squeeze().float().to(device)) + negative_penalty

                loss = input_weight * criterion(out.squeeze().to(device), local_labels.squeeze().float().to(device)) + negative_penalty
            elif mode == 2:
                # if mode 2, we add an additional regularization term
                loss = (input_weight) * criterion(out.squeeze().to(device), local_labels.squeeze().float().to(device)) + alpha * torch.square(weight) @ torch.square(eigenvalues)
                # loss = criterion(out.squeeze().to(device), local_labels.squeeze().float().to(device)) + alpha * torch.square(weight) @ torch.square(eigenvalues)

            acc = 0.01 * binary_acc(out.squeeze(), local_labels, seq2seq)
            # If we want to separate the subjects to high and low weight we could do it with the median
            if input_weight > median:
                counter_ge += 1
                avg_acc_ge += acc.item()
            elif input_weight <= median:
                counter_le += 1
                avg_acc_le += acc.item()
            avg_acc += acc.item()

            total_loss += loss + l1_regularizer(model, lambda_l1=0.001, weight_or_bias='weight')
            if counter % batch_size == 0 or counter == len(train_loader):
                total_loss.backward()
                optimizer.step()
                total_loss = 0

            avg_loss += (loss + l1_regularizer(model, lambda_l1=0.001, weight_or_bias='weight')).item()

        result = evaluate_last_timestep(model=model, val_loader=val_loader, device=device, criterion=criterion, lookup=lookup_val, weight=weight, basis=basis_val, centering=centering)

        if epoch % 5 == 0 or epoch==max_epochs-1 and counter > 0 and counter_ge > 0 and counter_le > 0:
            try:
                print(
                    "Epoch {} | Train Loss: {:.2f} | Train Acc : {:.4f} | Val loss: {:.4f} | Train Acc >0.5: {:.4f} | Train Acc <0.5: {:.4f} | Val Acc: {:.4f}".format(epoch, \
                        avg_loss / counter, avg_acc / counter, result['loss'], \
                        avg_acc_ge / counter_ge, avg_acc_le / counter_le, result['accuracy'])
                )
            except:
                print('Division by zero error!')

        if counter <= 0 or counter_ge <= 0 or counter_le <=0:
            print('hole')
        plot_loss.append((loss + l1_regularizer(model, lambda_l1=0.001, weight_or_bias='weight')).detach().numpy())
        plot_epoch.append(epoch)
        plot_train_acc.append(avg_acc / counter)
        plot_val_acc.append(result['accuracy'])
        plot_val_loss.append(result['loss'])
        scheduler.step()
    hist_data = centering + weight @ basis.T

    plot = {'epoch': plot_epoch, 'train_acc': plot_train_acc, 'val_acc': plot_val_acc, 'train_loss': plot_loss, 'val_loss': plot_val_loss}

    model_wrapper = ModelWrapper(model, h)
    model_wrapper.zero_grad()
    model_wrapper.eval()
    # Now use model_wrapper with GuidedBackprop
    gb = GuidedBackprop(model_wrapper)
    # gb = IntegratedGradients(model_wrapper)
    # dl = Saliency(model_wrapper)
    # dl = FeatureAblation(model_wrapper)

    # evaluate model from the last epoch of that fold
    result = evaluate_last_timestep_shap(model=model, val_loader=val_loader, device=device, criterion=criterion,
                                    lookup=lookup_val, weight=weight, basis=basis_val, explainer=gb, centering=centering)

    # Keep SHAP values of the fold for overall plots at the end of all folds
    value_list = [t.tolist() for t in result['shap_fold']]
    for SHAPs in value_list:
        SHAP_values_per_fold_deep.append(SHAPs)

    return model, h, plot, hist_data, weight, SHAP_values_per_fold_deep


    # def f(X):
    #     X = torch.tensor(X)
    #     h = model.init_hidden(batch_size=25*max_epochs)
    #     return model(X.float(), h.float())[0]
    # kernel_explainer = shap.KernelExplainer(f, data=local_batch.numpy())
    # deep_explainer = shap.DeepExplainer(model, data=(local_batch,h))
