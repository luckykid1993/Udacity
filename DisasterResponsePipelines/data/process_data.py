import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load and merge messages data with categories data
    
    inputs:
        messages_filepath (string): Filepath for csv file containing messages.
        categories_filepath (string): Filepath for csv file containing categories.
       
    outputs:
        df (dataframe):  A dataframe containing content of messages and categories.
    """

    # read messages from csv
    msg = pd.read_csv(messages_filepath)
    
    # read categories from csv
    ctgr = pd.read_csv(categories_filepath)
    
    # merge 2 datasets
    df = msg.merge(ctgr, how = 'inner', on = ['id'])
    
    return df


def clean_data(df):
    """Clean dataframe.
    
    input:
        df (dataframe): Dataframe that needed to clean.
       
    Returns:
        df_ret (dataframe): Cleaned dataframe.
    """
    
    # copy input data frame
    df_ret = df.copy()
    
    # create a dataframe of the 36 individual category columns
    categories =  df_ret.categories.str.split(';', expand = True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    category_colnames = row.transform(lambda x: x[:-2]).tolist()
    
    # rename the columns of `categories`
    categories.columns = category_colnames

    # keep only the last character of each string. For example, related-0 becomes 0, related-1 becomes 1. 
    # convert the string to a numeric value.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].transform(lambda x: x[-1])
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    
    # drop the original categories column from `df_ret`
    df_ret.drop('categories', axis = 1, inplace = True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df_ret = pd.concat([df_ret, categories], axis = 1)
    
    # remove child_alone column
    df_ret.drop('child_alone', axis = 1, inplace = True)
    
    # keep related = 0 or related = 1
    drop_index = df_ret[~df_ret['related'].isin([0,1])].index.tolist()
    df_ret.drop(drop_index, inplace = True)
    
    # Drop duplicates
    df_ret.drop_duplicates(inplace = True)
    
    return df_ret


def save_data(df, database_filename, replace_if_exists = True):
    """Save clean data into  SQLite database.
    
    inputs:
        df (dataframe): Dataframe containing cleaned data.
        database_filename (string): Filename of database.
        replace_if_exists (bool): replace exists data or not, default True
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    # replace exists data or not
    if replace_if_exists:
        df.to_sql('Msg_Category_Tbl', engine, index=False, if_exists='replace')
    else:
        df.to_sql('Msg_Category_Tbl', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterClean.db')


if __name__ == '__main__':
    main()