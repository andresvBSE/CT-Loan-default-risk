import pandas as pd 
import numpy as np

def missings_preprocessing(df_val, df_train):
    # Missing count
    df = df_val.copy()
    df['missing_count'] = df.isnull().sum(axis=1)
    
    # New columns with label 1 if had a missing value
    cols_to_dummy_missing = ['anual_payment', 'product_amount', 'accompained', 'vehicle_age', 'occupation', 'family_num', 
                             'external_score_1', 'external_score_2', 'external_score_3', 'num_apart_median', 
                             'area_basement_median', 'age_expl_median', 'age_building_median', 'area_common_median', 
                             'num_lifts_median', 'num_entries_median', 'max_floor_median', 'min_floor_median', 
                             'area_building_median', 'num_apart_habit_median', 'area_habit_median', 'num_apart_nohabit_median', 
                             'area_nohabit_median', 'type_building', 'total_area', 'wall_materials', 'emergency_exits', 
                             'num_petic_bureau_hour']
    
    for col in cols_to_dummy_missing:
        df[col+'_NA'] = df[col].isna().astype(int)
        
    # Delete columns for low importance and high proportion of missing values
    to_detele_cols = ['num_apart_nohabit_mode', 'vehicle_age', 'area_common_median', 'area_basement_mode', 'num_lifts_median',
                      'area_nohabit_average', 'area_habit_median', 'area_building_mode', 'age_building_average',
                      'age_building_mode', 'age_building_median', 'min_floor_mode', 'min_floor_median', 'total_area',
                      'min_floor_average', 'num_apart_habit_median', 'num_apart_habit_average', 'num_apart_nohabit_average', 
                      'num_apart_nohabit_median', 'area_common_mode', 'num_apart_habit_mode', 'area_building_median', 
                      'area_building_average', 'age_expl_median', 'area_basement_average', 'area_basement_median', 
                      'area_nohabit_median', 'area_nohabit_mode', 'num_lifts_average', 'num_apart_mode', 'num_apart_median', 
                      'num_entries_average', 'area_habit_average', 'max_floor_median', 'max_floor_average', 'age_expl_mode', 
                      'area_common_average']
    
    df.drop(to_detele_cols, axis=1, inplace=True)
    
    # Fill missings with 0
    fill_with_0 = ['num_lifts_mode', 'num_apart_average', 'num_entries_median', 'num_entries_mode','area_habit_mode',
                   'max_floor_mode', 'emergency_exits']
    
    df[fill_with_0]= df[fill_with_0].fillna(0)
    
    
    # Fill with unknow
    fill_with_unknow = ['wall_materials', 'occupation', 'type_building']
    df[fill_with_unknow] = df[fill_with_unknow].fillna('fill_with_unknow')
    
    # Fill with Other
    df['accompained'] = df['accompained'].replace(to_replace=['other_2', 'other_1', 'group', np.nan], value='other')
    
    # Fill external escores
    for iter_ in range(5):
        df['external_score_1'] = 0.1190 * df['external_score_2'] + 0.0504 * df['external_score_3'] - 0.0103 * df['age']/360
        df['external_score_2'] = 0.1751 * df['external_score_1'] + 0.2893 * df['external_score_3'] - 0.0067 * df['age']/360
        df['external_score_3'] = 0.0742 * df['external_score_1'] + 0.2895 * df['external_score_2'] - 0.0071 * df['age']/360

    df['no_external_score'] = np.where(df['external_score_1'].isna(), 1, 0)
    
    # Fix the age
    df['age'] = (df['age'] / 365) * -1
    df['age_group'] = pd.cut(df['age'], 
                         bins = [0, 25, 35, 45, 60, 70],
                         labels = ['<25', '25-34', '35-44', '45-59', '60>'])
    
    # Fill with values by group (mean values from the train dataset)
    fill_with_group_avg = ['age_expl_average', 'external_score_1', 'external_score_2', 'external_score_3', 
                           'num_petic_bureau_hour','num_petic_bureau_year', 'num_petic_bureau_quarter', 
                           'num_petic_bureau_month','num_petic_bureau_week', 'num_petic_bureau_day','product_amount', 
                           'anual_payment', 'age_mobilephone_days', 'family_num']
    
    group = ['gender', 'age_group', 'marital_status']

    for col in fill_with_group_avg:
        df[col] = df[col].fillna(df_train.groupby(group)[col].transform('mean'))

        # The rest with the global mean
        df[col] = df[col].fillna(df_train[col].mean())
    
    return df

def more_preprocessing(df_val):
    df = df_val.copy()
    # Age colums
    df['id_age'] = df['id_age'] * -1
    df['registry_age'] = df['registry_age'] * -1
    df['age_mobilephone_days'] = df['age_mobilephone_days'] * -1
    
    # New external scores
    df['external_score_min'] = df[['external_score_1', 'external_score_2', 'external_score_3']].min(axis=1)
    df['external_score_max'] = df[['external_score_1', 'external_score_2', 'external_score_3']].max(axis=1)
    df['external_score_median'] = df[['external_score_1', 'external_score_2', 'external_score_3']].median(axis=1)
    df['external_score_mean'] = df[['external_score_1', 'external_score_2', 'external_score_3']].mean(axis=1)
    
    return df
    