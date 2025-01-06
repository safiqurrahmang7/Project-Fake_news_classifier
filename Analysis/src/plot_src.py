import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from abc import ABC, abstractmethod
from wordcloud import WordCloud

class eda_plots(ABC):
    @abstractmethod
    def plotting(df:pd.DataFrame):
        pass

class plot_duplicates(eda_plots):


    def plotting(df:pd.DataFrame):
        # Count the duplicates for each column
        text_duplicates = df['text'].duplicated().sum()

        # Create a DataFrame for plotting
        data_text = [text_duplicates, len(df) - text_duplicates]

        labels = ['Duplicates', 'Unique']

        # Text column pie chart
        plt.figure(figsize=(10,8))
        plt.pie(data_text, labels=labels, autopct='%1.1f%%',explode=(0,0.1), startangle=90, colors=['#ff9999','#66b3ff'])
        plt.title('Duplicates in Text Column')

        plt.show()

class plot_missing_values(eda_plots):

    def plotting(df:pd.DataFrame):

       # Missing values heatmap
        
        sns.heatmap(df.isnull(),cbar = False, cmap = 'viridis')
        plt.title('Distribution of Missing Values')
        plt.show()

        plt.show()

class plot_wordcloud(eda_plots):

    def plotting(df:pd.DataFrame):

        # Create a wordcloud for the text column
        text = " ".join(df['text'])

        wordcloud = WordCloud(width=1000,height=500,background_color='black').generate(text)

        plt.Figure(figsize=(10,5))
        plt.imshow(wordcloud,interpolation='bilinear')
        plt.axis('off')
        plt.show()

class plot_distribution(eda_plots):

    def __init__(self,target:str):
        self.target = target

    def plotting(self,df:pd.DataFrame):

        # Distribution of the target column
        plt.figure(figsize=(10,8))
        plt.pie(df[self.target].value_counts(), labels=['Real', 'Fake'], autopct='%1.1f%%',explode=(0,0.1), startangle=90, colors=['green','red'])
        plt.title('Distribution of the Target Column')
        plt.show()


class plotter:

    def __init__(self,plot = eda_plots):
        
        self.plotter = plot

    def set_plotter(self,plot = eda_plots):

        self.plotter = plot

    def apply_plotter(self,df:pd.DataFrame,column:str=None):

        return self.plotter.plotting(df)
        

        


