from dataframe_utils import *


def load_longitudinal_tabular_data(input_path, write_path, quick=True, write_csv=True):

    # We can save a 'processed' version of the input data to load them quickly each time
    if quick:
        input_feats = read_csv(input_path, separator=',')
    else:
        input_data = read_csv(input_path, separator=',')

        input_feats = tabular_data_processing(input_data)

        if write_csv:
            input_feats.to_csv(os.path.join(write_path, 'processed.csv'), index=False)

    return input_feats

def tabular_data_processing(input_data):

    # Do not use confounders as input variables to the model
    input_data = drop_confounders(input_data, age=False)

    # Some specific processing required for the stroop input variables
    # add the features and then drop the columns we summed
    input_data = sum_stroop(input_data)

    # drop the features we don't need
    input_data = drop_features(input_data, age=False)

    # Rename the new constructs that we will predict to make it easier to remember which is which
    input_data = rename_constructs(input_data)

    input_data = remove_moderate_and_empty(input_data)
    input_data = make_cahalan_binary(input_data)
    input_data, already_binging = remove_non_control(input_data)

    # Fill out missing values per subject with nearest visit values
    input_feats = fill_out_missing_values_nearest(input_data)
    already_binging = fill_out_missing_values_nearest(already_binging)

    # For some subjects we only have one visit therefore we still need the columnn mean/mode
    # to fill out the missing values
    input_feats = fill_out_missing_values(input_feats)
    already_binging = fill_out_missing_values(already_binging)

    # Turn variables such as sex from 'F' and 'M' to 0 and 1
    input_feats = text_to_codes(input_feats)
    already_binging = text_to_codes(already_binging)

    return input_feats, already_binging


# Find common subjects among imaging and tabular input variables longitudinal
def common_subjects_longitudinal(avg_visit_data, avg_visit_imaging_scores):

    common = set(avg_visit_imaging_scores['subject']).intersection(set(avg_visit_data['subject']))

    common_imaging_scores = avg_visit_imaging_scores.loc[avg_visit_imaging_scores['subject'].isin(common)]

    common_scores = pd.merge(avg_visit_data, common_imaging_scores, on=['subject', 'visit'])

    return common_scores


def load_longitudinal_with_imaging(input_path, write_path, input_imaging_path_fa, input_imaging_path_t1=None, input_imaging_path_rsf1=None, 
    input_imaging_path_rsf2= None, quick=True, write_csv=False):

    # Load imaging scores
    imaging_scores_fa = read_csv(input_imaging_path_fa, separator=',')
    imaging_scores = imaging_scores_fa

    if input_imaging_path_t1 != None:
        imaging_scores_t1 = read_csv(input_imaging_path_t1, separator=',')
        imaging_scores = pd.merge(imaging_scores_fa, imaging_scores_t1, on=['subject', 'visit'])

    if input_imaging_path_rsf1 != None:
        imaging_scores_rsf = read_csv(input_imaging_path_rsf1, separator=',')
        imaging_scores = pd.merge(imaging_scores, imaging_scores_rsf, on=['subject', 'visit'])

    if input_imaging_path_rsf2 != None:
        imaging_scores_rsf = read_csv(input_imaging_path_rsf2, separator=',')
        imaging_scores = pd.merge(imaging_scores, imaging_scores_rsf, on=['subject', 'visit'])

    # We can save a 'processed' version of the input data to load them quickly each time
    if quick:
        input_feats = read_csv(input_path, separator=',')
    else:
        input_data = read_csv(input_path, separator=',')

        input_feats, additional = tabular_data_processing(input_data)

    # Merge the two sets of information to the overall data
    input_feats = common_subjects_longitudinal(input_feats, imaging_scores)
    additional = common_subjects_longitudinal(additional, imaging_scores)

    if write_csv:
        input_feats.to_csv(os.path.join(write_path, 'processed_with_imaging_t1.csv'), index=False)

    return input_feats, additional

def load_forecast_label_data(input_path, write_path, prossesed=True, write_csv=True):
    if prossesed:
        input_feats = read_csv(input_path, separator=',')
    else:
        input_feats = read_csv(input_path, separator=',')

        input_feats = remove_below_18(input_feats)

        input_feats = remove_22(input_feats)
     
        input_feats = remove_moderate_and_empty(input_feats, column_name='cahalan')

        input_feats = make_cahalan_binary(input_feats, column_name='cahalan')

        input_feats = input_feats[['subject', 'cahalan']]

        input_feats = input_feats.groupby(['subject'])['cahalan'].max()

        input_feats = pd.DataFrame({'subject':input_feats.index, 'cahalan_score':input_feats.values})

        if write_csv:
            input_feats.to_csv(os.path.join(write_path, 'forecast_label.csv'), index=False)
        return input_feats

def load_forecast_data(input_path, target_path, write_path, image_path_fa):
    target = load_forecast_label_data(target_path, write_path, prossesed=False, write_csv=False)
    data, additional = load_longitudinal_with_imaging(input_path, write_path, image_path_fa, input_imaging_path_t1=None, \
        input_imaging_path_rsf1=None, input_imaging_path_rsf2=None, quick=False, write_csv=False)

    common = set(target['subject']).intersection(set(data['subject']))
    common_1 = set(target['subject']).intersection(set(additional['subject']))

    target_0 = target.loc[target['subject'].isin(common)]
    target_1 = target.loc[target['subject'].isin(common_1)]

    data = data.loc[data['subject'].isin(common)]
    additional = additional.loc[additional['subject'].isin(common_1)]

    data = pd.merge(data, target_0, on=['subject'])
    additional = pd.merge(additional, target_1, on=['subject'])

    data.to_csv(os.path.join(write_path, 'processed_with_imaging_fa_merged.csv'), index=False)
    additional.to_csv(os.path.join(write_path, 'additional_with_imaging_fa_merged.csv'), index=False)


if __name__ == '__main__':
    # image_path_t1 = './T1_freesurfer_aseg.csv'
    # image_path_rsf1 = './rsfmri_corr.csv'
    # image_path_rsf2 = './rsfmri_part_corr.csv'
    # load_longitudinal_with_imaging(input_path, write_path, image_path_fa, image_path_t1, 
    #    input_imaging_path_rsf1=image_path_rsf1, input_imaging_path_rsf2=image_path_rsf2, quick=False, write_csv=True)
    write_path = './forecast_data'
    target_path = 'source_data/cahalan_plus_drugs.csv'
    input_path = 'source_data/full_per_visit_data_2021-03-26.csv'
    image_path_fa = 'image_data/DTI_fa.csv'

    load_forecast_data(input_path, target_path, write_path, image_path_fa)