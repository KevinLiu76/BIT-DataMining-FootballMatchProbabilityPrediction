import datetime as dt
import tensorflow as tf
import pandas as pd

def data_convert(df, feature: str, timesteps: int, mask:float, historical_features):
    df_ = df.copy()
    if feature not in historical_features:
        features = [feature]
    else:
        features = [f'{feature}_{i}' for i in range(1, timesteps + 1)][::-1]
    df_ = df_[['id'] + features]
    df_ = df_.fillna(mask)
 
    series = df_.set_index('id').stack().reset_index(level=1)[0].rename(feature)
    return series

def data_processing():
    mask = -999.0
    timesteps = 10
    historical_features = [
        # home based
        'home_team_history_goal',
        'home_team_history_opponent_goal',
        'home_team_history_is_play_home', 
        'home_team_history_rating',
        'home_team_history_opponent_rating',
        'home_team_history_match_days_ago',
        
        # away based
        'away_team_history_goal', 
        'away_team_history_opponent_goal',
        'away_team_history_rating',
        'away_team_history_opponent_rating',
        'away_team_history_match_days_ago'
    ] 
    
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    
    for col in train.filter(regex='date', axis=1).columns:
        train[col] = pd.to_datetime(train[col])
        test[col] = pd.to_datetime(test[col])
    
    # date based features
    for i in range(1, 11):
        train[f'home_team_history_match_days_ago_{i}'] = (train['match_date'] - train[f'home_team_history_match_date_{i}']).dt.days
        train[f'away_team_history_match_days_ago_{i}'] = (train['match_date'] - train[f'away_team_history_match_date_{i}']).dt.days
        test[f'home_team_history_match_days_ago_{i}'] = (test['match_date'] - test[f'home_team_history_match_date_{i}']).dt.days
        test[f'away_team_history_match_days_ago_{i}'] = (test['match_date'] - test[f'away_team_history_match_date_{i}']).dt.days
    
    # remove two matchs with possible error
    train = train[train['home_team_name'] != train['away_team_name']].reset_index(drop=True)

    validation_split = dt.datetime.strptime(
    '2021-05-01 00:00:00', '%Y-%m-%d %H:%M:%S') - dt.timedelta(days=70, hours=23, minutes=15)

    # maps
    target2int = {'away': 0, 'draw': 1, 'home': 2}

    # encode target
    train['target_int'] = train['target'].map(target2int)

    X_test = test.copy()

    # split train/val
    X_train = train[train['match_date'] <= validation_split].reset_index(drop=True)
    X_val = train[train['match_date'] > validation_split].reset_index(drop=True)
    # preprocess
    features_pattern = '_[0-9]|'.join(historical_features) + '_[0-9]'
    features_to_preprocess = train.filter(regex=features_pattern, axis=1).columns.tolist()
    X_train[['id', 'league_name', 'target_int'] + features_to_preprocess].to_csv("../data/X_train.csv", index=False, sep=',')
    X_val[['id', 'league_name', 'target_int'] + features_to_preprocess].to_csv("../data/X_val.csv", index=False, sep=',')
    X_test[['id', 'league_name'] + features_to_preprocess].to_csv("../data/X_test.csv", index=False, sep=',')

    X_train.to_csv("../data/X_train.csv", index=False, sep=',')
    X_val.to_csv("../data/X_val.csv", index=False, sep=',')
    X_test.to_csv("../data/X_test.csv", index=False, sep=',')
    
if __name__ == "__main__":
    data_processing()
