import pandas as pd
from src.text_preprocessing import text_preprocessing_spacy, text_processor


df = pd.read_csv('D:/Project-Fake_news_classifier/ProcessedData/fake_news_eda.csv')


# Initializing the text processor
processor = text_processor(text_preprocessing_spacy)

# Applying the text processor
df['text'] = df['text'].apply(lambda text: processor.apply_processor(text))

#saving the processed data
df.to_csv('D:/Project-Fake_news_classifier/ProcessedData/fake_news_processed.csv',index=False)
