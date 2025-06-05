import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# ensure the "logs" directory exits
log_dir = "logs"
os.makedirs(log_dir, exist_ok = True)

# logging configuration
logger = logging.getLogger("data_preprocessing")
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    """
    Transforms the input text by converting it to lowercase, tokenizing, removing stopwords and punctuaions, and stemming.
    """

    ps = PorterStemmer()
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text = [ps.stem(word) for word in text]
    
    return " ".join(text)

def preprocess_df(df: pd.DataFrame, text_column = 'text', target_column = 'target'):
    """
    Preprocesses the DataFrame by encoding the target column, removing duplicates, and transforming text column. 
    """

    try:
        logger.debug(f"Starting preprocessing for DataFrame")
        # Encode the target column
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug(f"Target column encoded")  

        # Remvove dulicate rows
        df = df.drop_duplicates(keep = 'first')
        logger.debug(f"Duplicates removed")

        # Apply text transformation to specified text column
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug(f"Text column transformed")
        return df
    except KeyError as e:
        logger.error(f"Column not found: {e}")
    except Exception as e:
        logger.error(f"Error during normalization: {e}")

def main(text_column = "text", target_column = "target"):
    """
    Main function to load raw data, preprocess it, and save the preprocessed data
    """
    try:
        # fetch the data from data/raw
        train_data = pd.read_csv("./data/raw/train.csv")
        test_data = pd.read_csv("./data/raw/test.csv")
        logger.debug(f"Data loaded properly")

        # Transform data
        train_processed_data = preprocess_df(train_data, text_column, target_column)
        test_processed_data = preprocess_df(test_data, text_column, target_column)

        # store the data inside data/processed
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok = True)

        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index = False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index = False)
        logger.debug(f"Processed data saved to {data_path}")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except pd.errors.EmptyDataError as e:
        logger.error(f"Data not found: {e}")
    except Exception as e:
        logger.error(f"Failed to complete data transformation process: {e}")
        print(f"Error : {e}")

if __name__ == '__main__':
    main()
