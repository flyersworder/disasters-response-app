import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Returns a dataframe that combines both disaster response messages and categories.

            Parameters:
                    messages_filepath (str): File path of the message data
                    categories_filepath (str): File path of the categories data

            Returns:
                    dataframe: A pandas dataframe that combines both
    '''
    # read csv files into a dataframe
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge these dataframes both on the id
    df = pd.merge(messages, categories, how='inner', on='id')
    
    return df

def clean_data(df):
    '''
    Returns a cleaned dataframe.

            Parameters:
                    df: An input dataframe that is not cleaned

            Returns:
                    dataframe: A dataframe that is cleaned
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.str[:-2].tolist()
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
    # drop the original categories column from `df`
    df.drop(columns='categories', inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)  
    
    # dro outliers: the `related` category contains value of 2
    df = df[df['related']!=2]
    
    return df

def save_data(df, database_filename):
    '''
    Save the dataframe into a database.

            Parameters:
                    df: A pandas dataframe
                    database_filename (str): File name of the database

            Returns:
                    None
    '''  
    # create a sql engine
    engine = create_engine('sqlite:///' + database_filename)
    
    # save the dataframe into the database
    df.to_sql('messages', engine, if_exists='replace', index=False)

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