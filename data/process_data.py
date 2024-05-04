# Import necessary libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories datasets
    """
    # Load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # Load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # Merge datasets
    df = pd.merge(messages, categories, on='id')
    
    return df

def clean_data(df):
    """
    Clean the merged dataset
    """
    # Split categories into separate category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # Use the first row to extract a list of new column names for categories
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    # Convert category values to binary representation
    for column in categories:
        # Convert values to binary
        categories[column] = categories[column].apply(lambda x: 1 if int(x[-1]) > 0 else 0)
    
    # Drop the original categories column from df
    df.drop('categories', axis=1, inplace=True)
    
    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    return df

def save_data(df, database_filename):
    """
    Save the cleaned data to an SQLite database
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('disaster_response', engine, index=False, if_exists='replace')

def main():
    """
    Main function to execute the ETL pipeline
    """
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
              'datasets as well as the filepath of the database to save the cleaned data '\
              'to. \n\nExample: python process_data.py disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()
