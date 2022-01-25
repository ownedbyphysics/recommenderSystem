import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('wordnet')
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from sklearn.metrics.pairwise import linear_kernel


settings = {'data': pd.read_csv('dedacted_sku_dataset.csv', index_col = 0)}


class description_similarity():
    """A module that computes the similarity matrix in a dataset that consists
       of a unique key (SKU here) of a product and the text-description that the
       site provided to every corresponding product. The methods of the class
       process and clean the text, produces the similarity matrix, saves to a 
       location and loads it again. The final step is the recommend method
       that takes as parameter a test SKU sample and the number of the top 
       similar products the user want to be displayed.
       
       Args:
        settings: an initialization dictionary of the form:
        >>> settings = {
        ...     'data': df,
        }
    
    Returns:
        A dataframe of chosen length with most similar SKUs
  
      Example:
     >>> p = description_similarity(settings)
     >>> p.data_read()
     >>> p.description_matrix()
     >>> p.save('mySimilarityMatrix', matrix)
     >>> p.load('mySimilarityMatrix', matrix)
     >>> p.load('mySimilarityMatrix', matrix)
     >>> p.recommendations("watch_5", 10)
     """
    
    def __init__(self, settings):
        self.settings = settings
       
    
    def data_read(self, df_to_process=None):
        """Read the default dataset or any other users dataset.
           The dataset should contain an SKU column to be used as 
           unique index and a text description column

        Args:
            database(optional): By default the DataFrame comes 
            from the setting dictionary
        """ 
        try:
            if isinstance(df_to_process, pd.DataFrame) and not df_to_process.empty:
                df = df_to_process
            else:
                df = self.settings['data']
                
            df.reset_index(inplace=True)
            
            if 'sku' not in df.columns:
                raise AttributeError("the given dataframe should contain the sku column")
            elif 'description' not in df.columns or 'description' not in df.columns:
                raise AttributeError("the given dataframe should contain the description column")
        except Exception as other_exception:
            print(other_exception)       
        
        df['description'] = df['description'].apply(self.cleaningText)
        self.df = df
       
   
    def cleaningText(text, stem=False):
        """A method responsible for cleaning the text using a plethora 
           of methods.

        Args:
            text: the string based text ought to be cleaned
            stem: bool variable to determine if the stem or the lemma
                  procedure will be the final cleaning step. If false, 
                  then the lemmatizer will be the initialized and vice verca.
        """ 
        stop_words = stopwords.words('english')
        string.punctuation = string.punctuation + '£' +'’' + '...'
        text = text.lower() # Lowercase 
        text = text.strip() # Remove leading/trailing whitespace
        text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text) # Remove punctuation
        text = re.sub('\s+', ' ', text) # Remove extra space and tabs
        text = re.compile('<.*?>').sub('', text) # Remove any HTML tags/markups

        #import pdb; pdb.set_trace()
        filtered_sentence=[]
        # Tokenize the sentence
        words = word_tokenize(text)
        for w in words:
            if w not in stop_words:
                filtered_sentence.append(w)
        text = " ".join(filtered_sentence)

        if stem:
            snow = SnowballStemmer('english')
            stemmed_sentence = []
            # Tokenize the sentence
            words = word_tokenize(text)
            for w in words:
                # Stem the word/token
                stemmed_sentence.append(snow.stem(w))
            text = " ".join(stemmed_sentence)
        else:
            lemmatizer = WordNetLemmatizer()
            # This is a helper function to map NTLK position tags
            def get_wordnet_position(tag):
                if tag.startswith('J'):
                    return wordnet.ADJ
                elif tag.startswith('V'):
                    return wordnet.VERB
                elif tag.startswith('N'):
                    return wordnet.NOUN
                elif tag.startswith('R'):
                    return wordnet.ADV
                else:
                    return wordnet.NOUN

            lemmatized_sentence = []
            # Tokenize the sentence
            words = word_tokenize(text)
            # Get position tags
            word_pos_tags = nltk.pos_tag(words)
            # Map the position tag and lemmatize the word/token
            for idx, tag in enumerate(word_pos_tags):
                lemmatized_sentence.append(lemmatizer.lemmatize(tag[0], get_wordnet_position(tag[1])))
            text = " ".join(lemmatized_sentence)

        return text
    
    def description_matrix(self):
        """In this method the similarity matrix is computed.
           One can choose of different methods to produce it
           even though the differences are mostly visible when
           the dataset lenght is very large
        """ 
        descriptions = [item for item in self.df['description']]
        self.sku_list = self.df['sku']
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(descriptions)
        
        # pairwise similarity be matrices multiplication
        # without using any extenal libraries
        pairwise_similarity = tfidf_matrix * tfidf_matrix.T
        pairwise_similarity = pairwise_similarity.A
        
        # another method is the linearl kernel to compute similarities
        #pairwise_similarity= linear_kernel(tfidf_matrix)
        
        # last way is to use the cosine similarity external library
        #pairwise_similarity = cosine_similarity(tfidf)
        
        similarity_df = pd.DataFrame(pairwise_similarity)
        similarity_df = similarity_df.set_index(self.sku_list)
        similarity_df.columns = self.sku_list
        self.similarity_df = similarity_df
        display(similarity_df.head())

    def save(self, fileName, specific_path=False):
        if specific_path:
            self.similarity_df.to_csv(specific_path + '/' + str(fileName) + '.csv')
        else:
            self.similarity_df.to_csv(str(fileName) +'.csv')
        


    def load(self, fileName, from_path=False):
        if from_path:
            self.similarity_df = pd.read_csv(from_path + '/' + fileName + '.csv')
            self.similarity_df = self.similarity_df.set_index(self.sku_list)
        else:
            self.similarity_df = pd.read_csv(fileName + '.csv')
            self.similarity_df = self.similarity_df.set_index(self.sku_list)
        return self.similarity_df.head()
        
    
    def recommendations(self, testSKU, number_of_similar):
        final = self.similarity_df[[testSKU]]
        final = final.nlargest(number_of_similar+1, testSKU)
        final.drop(index = testSKU, inplace = True)
        
        return final       