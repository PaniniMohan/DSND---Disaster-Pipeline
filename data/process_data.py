#This is the step - 1 of the DSND project Disaster response pipeline'
#To execute this code, you simply need to type the following statement in the command window -
#python ./data/process_data.py ./data/disaster_messages.csv ./data/disaster_categories.csv DisasterResponse.db'


#All import statements necessary for the execution'

import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

#LOad function to import the CSV file

def load_data(messages_location, categories_location):
    """
    The function takes two csv files as the inputs and merges them as a dataframe - returning it to the user
    
    """
    messages = pd.read_csv(messages_location)
    categories = pd.read_csv(categories_location)
    df = pd.merge(messages,categories,on='id')
    return df 

#Data manipulation function to import the CSV file

def clean_data(df):
    """
    This function's objective is to clean up the categories column and replace it with clean text
    """
    categories = df.categories.str.split(';',expand=True)
    categories.head()
    row = categories.loc[1]
    
    # select the first row of the categories dataframe
    row = categories.loc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x:x[:-2])
    print(category_colnames)
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(np.int)
    #Drop the previous categories
    
    df = df.drop('categories',axis=1)
    df = pd.concat([df,categories],axis=1)
    
    print('Repeating rows: {} There are total of {} '.format(df.duplicated().sum(),df.shape[0]))
    df = df.drop_duplicates()
    print('There are total of {} after removing duplicates'.format(df.shape[0]))
    
    return df


def save_data(df, database_filename):
    """
    Save Data function
    
    Arguments:
        df -> Clean data Pandas DataFrame
        database_filename -> database file (.db) destination path
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('df', engine, index=False)
    pass  

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
              'DisasterResponse.db')


if __name__ == '__main__':
    main()