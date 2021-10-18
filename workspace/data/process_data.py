import sys
import os
import pandas as pd
import numpy as np
import math
import json
import datetime
from sqlalchemy import create_engine


def load_data(portfolio_filepath, profile_filepath,transcript_filepath):

    # read in the json files
    portfolio = pd.read_json(portfolio_filepath, orient='records', lines=True)
    profile = pd.read_json(profile_filepath, orient='records', lines=True)
    transcript = pd.read_json(transcript_filepath, orient='records', lines=True)

    return portfolio,profile,transcript

def clean_data_portfolio(portfolio):

    # create new portfolio data set copy from portfolio
    portfolio_new=portfolio.copy()

    # rename id column to offer_id
    portfolio_new.rename(columns={'id':'offer_id'},inplace=True)

    # change duration column from days to hours because the time in transaction is hours
    portfolio_new['duration'] = portfolio_new['duration']*24

    # rename duration column to duration_hours
    portfolio_new.rename(columns={'duration':'duration_hours'},inplace=True)

    # get unique channels
    channels=[]
    for i in range(portfolio_new.shape[0]):
        channels.extend(portfolio_new.channels[i])
    channels=set(channels)

    # create each channel as column and assign value
    for i in channels:
        portfolio_new[i]=portfolio_new['channels'].apply(lambda x: 1 if i in x else 0)

    # drop original channels column
    portfolio_new.drop('channels', axis=1, inplace=True)

    # get dummies for offer_type
    offer_type=pd.get_dummies(portfolio_new['offer_type'])

    # merge offer_type in portfolio_new dataframe
    portfolio_new=pd.concat([portfolio_new, offer_type], axis=1, sort=False)

    # drop off_type column
    portfolio_new.drop('offer_type', axis=1, inplace=True)

    # replace the offer_id by num ids
    labels_offer_id = portfolio_new['offer_id'].astype('category').cat.categories.tolist()
    offer_id_num_map = {'offer_id' : {k: v for k,v in zip(labels_offer_id,list(range(1,len(labels_offer_id)+1)))}}
    portfolio_new.replace(offer_id_num_map, inplace=True)

    return portfolio_new

def clean_data_profile(profile):

    # create a copy from profile dataset
    profile_new = profile.copy()

    # drop null values in rows which should be age==118, seems outliers
    profile_new.dropna(inplace=True)

    # rename id column customer_id
    profile_new.rename(columns={'id':'customer_id'},inplace=True)

    # change became_member_on data type from int to datetime format
    profile_new['became_member_on'] = pd.to_datetime(profile_new['became_member_on'], format = '%Y%m%d')

    # calculate the number of days since the user is a memeber of starbucks
    profile_new['memberdays'] = datetime.datetime.today().date() - profile_new['became_member_on'].dt.date
    profile_new['memberdays'] = profile_new['memberdays'].dt.days

    # remove became_member_on column
    profile_new.drop('became_member_on', axis=1, inplace=True)

    # replace the gender with numerical label
    labels_gender = profile_new['gender'].astype('category').cat.categories.tolist()
    gender_num_map = {'gender' : {k: v for k,v in zip(labels_gender,list(range(1,len(labels_gender)+1)))}}
    profile_new.replace(gender_num_map, inplace=True)

    return profile_new

def clean_data_transcript(transcript):

    # create a copy from transcript dataset
    transcript_new = transcript.copy()

    # rename person column to customer_id
    transcript_new.rename(columns={'person':'customer_id'},inplace=True)

    # drop all the transaction events, only left offer events
    transcript_new.drop(transcript_new[transcript_new['event']=='transaction'].index,inplace=True)

    # get offer_id from value column
    transcript_new['offer_id']=transcript_new['value'].apply(lambda x: x['offer id'] if ('offer id' in x) else x['offer_id'])

    # drop value column
    transcript_new.drop(['value','time'],axis=1, inplace=True)

    # replace the offer_id by num ids
    labels_offer_id = transcript_new['offer_id'].astype('category').cat.categories.tolist()
    offer_id_num_map = {'offer_id' : {k: v for k,v in zip(labels_offer_id,list(range(1,len(labels_offer_id)+1)))}}
    transcript_new.replace(offer_id_num_map, inplace=True)

    # split three kinds event to different dataframe
    offer_received=transcript_new[transcript_new.event=='offer received'] .drop_duplicates()
    offer_viewed=transcript_new[transcript_new.event=='offer viewed'][['offer_id','customer_id']].assign(viewed_count=1)
    offer_completed=transcript_new[transcript_new.event=='offer completed'][['offer_id','customer_id']].assign(completed_count=1)

    # get offer viewed and completed count
    offer_viewed=offer_viewed.groupby(['offer_id','customer_id'])['viewed_count'].count().reset_index(name='viewed_count')
    offer_completed=offer_completed.groupby(['offer_id','customer_id'])['completed_count'].count().reset_index(name='completed_count')

    # merge offer_viewed and completed with offer received
    offer_received=offer_received.merge(offer_viewed,how ='left', on = ['customer_id','offer_id'])
    offer_received=offer_received.merge(offer_completed,how ='left', on = ['customer_id','offer_id'])

    # fill NA value with 0
    offer_received.fillna(0,inplace=True)

    # create completed column based on completed_count
    offer_received['completed']=offer_received['completed_count'].apply(lambda x: x if x<=1 else 1)

    return offer_received

def merge_data(offer_received,portfolio_new,profile_new):

    # merge transcript_new dataset with portfolio_new on offer_id
    user_offer_matrix =offer_received.merge(portfolio_new,how='left',on='offer_id')

    # merge clean_df dataset with profile_new on customer_id
    user_offer_matrix = user_offer_matrix.merge(profile_new,how ='left', on = 'customer_id')

    # remove all the NA values
    user_offer_matrix.dropna(inplace=True)

    return user_offer_matrix

def save_data(user_offer_matrix, database_filename):
    if os.path.exists(database_filename):
        os.remove(database_filename)
    #engine = create_engine('sqlite:///../data/{}'.format(database_filename))
    engine = create_engine('sqlite:///{}'.format(database_filename))
    user_offer_matrix.to_sql('user_offer_matrix', engine, index=False)


def main():
    if len(sys.argv) == 5:

        portfolio_filepath, profile_filepath,transcript_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    portfolio: {}\n    profile: {}\n    transcript: {}'
              .format(portfolio_filepath, profile_filepath,transcript_filepath))
        portfolio,profile,transcript = load_data(portfolio_filepath, profile_filepath,transcript_filepath)

        print('Cleaning data...')

        portfolio_new= clean_data_portfolio(portfolio)
        portfolio_new.to_pickle('data/portfolio.pkl')
        profile_new= clean_data_profile(profile)
        offer_received= clean_data_transcript(transcript)

        print('Merging data...')
        user_offer_matrix=merge_data(offer_received,portfolio_new,profile_new)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(user_offer_matrix, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the portfolio,profile,transcript '\
              'datasets as the first, second and third argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the fourth argument. \n\nExample: python process_data.py '\
              'portfolio.json profile.json transcript.json '\
              'user_offer_matrix.db')


if __name__ == '__main__':
    main()
