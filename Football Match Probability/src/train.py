import os
import random
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import argparse
from evaluate import evaluate
from sklearn import preprocessing
from model import create_LSTM_model 
from data_processing import data_convert

    
def train(args):
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
    
    train = pd.read_csv(args.train_dataset_path)
    val = pd.read_csv(args.val_dataset_path)

    
    # preprocess
    features_pattern = '_[0-9]|'.join(historical_features) + '_[0-9]'
    features_to_preprocess = train.filter(regex=features_pattern, axis=1).columns.tolist()

    # this Scaler removes the median and scales the data according to the quantile range (defaults to IQR: Interquartile Range)
    scaler = preprocessing.RobustScaler()
    X_train_pre = pd.DataFrame(scaler.fit_transform(train[features_to_preprocess]), columns=features_to_preprocess)
    X_train = pd.concat([train[['id', 'league_name', 'target_int']], X_train_pre], axis=1)
    X_val_pre = pd.DataFrame(scaler.transform(val[features_to_preprocess]), columns=features_to_preprocess)
    X_val = pd.concat([val[['id', 'league_name', 'target_int']], X_val_pre], axis=1)



    # create targets
    y_train = data_convert(X_train, 'target_int', timesteps=timesteps, mask = mask, historical_features = historical_features).values.reshape(-1, 1)
    y_val = data_convert(X_val, 'target_int', timesteps=timesteps, mask = mask, historical_features = historical_features).values.reshape(-1, 1)

    # create historical features
    X_train = pd.concat([
        data_convert(X_train, feature=feature, timesteps=timesteps, mask = mask, historical_features = historical_features) for feature in historical_features
    ], axis=1).values.reshape(-1, timesteps, len(historical_features))
    X_val = pd.concat([
        data_convert(X_val, feature=feature, timesteps=timesteps, mask = mask, historical_features = historical_features) for feature in historical_features
    ], axis=1).values.reshape(-1, timesteps, len(historical_features))


    early_stopping_patience = args.epochs // 10
    es = tf.keras.callbacks.EarlyStopping(
        patience=early_stopping_patience,
        restore_best_weights=True,
        verbose=1
    )
    rlrop = tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.8,
        patience=early_stopping_patience // 2,
        verbose=1
    )

    model = create_LSTM_model(X_train,mask,args.classes) 
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

    h = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=[
            es,
            rlrop
        ]
    )
    tf.keras.models.save_model(model, args.save_model_dir + '/model.h5')
    plt.plot(h.history['loss'], label='Train')
    plt.plot(h.history['val_loss'], label='Val')
    plt.legend()
    
    evaluate(model, X_val, y_val)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed")
    # data config
    parser.add_argument("--train_dataset_path", type=str, default='../data/X_train.csv',
                         help="path of train_dataset")
    parser.add_argument("--val_dataset_path", type=str, default='../data/X_val.csv',
                         help="path of val_dataset")
    parser.add_argument("--save_model_dir", type=str,
                        help="model dir to save models")
    parser.add_argument("--classes", type=int, default=3,
                        help="the classes of results")
    parser.add_argument("--epochs", type=int, default=100,
                        help="epochs of training")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="batch size of training")
    args = parser.parse_args()
    
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    os.makedirs(args.save_model_dir, exist_ok=True)
    
    train(args)
