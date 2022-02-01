{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import scipy\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "from scipy.spatial.distance import pdist,squareform\n",
    "\n",
    "settings = {'data':pd.read_csv('dedacted_sku_brand_features.csv'),\n",
    "            'target':'SKU'}\n",
    "\n",
    "class feature_similarity():\n",
    "    \"\"\"A module that computes the similarity matrix in a dataset that consists of\n",
    "       several watch-related features such us brand, color, price etc and\n",
    "       of a key (SKU here) that identifies uniquely every product. The methods of the class\n",
    "       reads the data, label encode the non numerical columns and standardize the values to \n",
    "       produce the similarity matrix. Moreover there are methods that save and loads the produced\n",
    "       similarity matrix. The final step is the recommend method\n",
    "       that takes as parameter a test SKU sample and the number of the top \n",
    "       similar products the user want to be displayed.\n",
    "       \n",
    "       Args:\n",
    "        settings: an initialization dictionary of the form:\n",
    "        >>> settings = {\n",
    "        ...     'data': df,\n",
    "        }\n",
    "    \n",
    "    Returns:\n",
    "        A dataframe of chosen length with the most similar SKUs\n",
    "  \n",
    "      Example:\n",
    "     >>> p = description_similarity(settings)\n",
    "     >>> p.data_preprocessing('SKU')\n",
    "     >>> p.matrix()\n",
    "     >>> p.save('featureSimilarityMatrix', matrix)\n",
    "     >>> p.load('featureSimilarityMatrix', matrix)\n",
    "     >>> p.recommendation(\"watch_5\", 10)\n",
    "     \"\"\"\n",
    "    \n",
    "    def __init__(self, settings):\n",
    "        self.settings = settings\n",
    "        \n",
    "    def data_preprocessing(self, target, df_to_process=False):\n",
    "        \"\"\"A method that processes the database and\n",
    "           encodes categorical columns and scales the values to a [0,1] map\n",
    "\n",
    "        Args:\n",
    "            database(optional): By default the DataFrame comes \n",
    "            from the setting dictionary\n",
    "        \"\"\"\n",
    "        try:\n",
    "            if isinstance(df_to_process, pd.DataFrame) and not df_to_process.empty:\n",
    "                df = df_to_process\n",
    "            else:\n",
    "                df = self.settings['data'] \n",
    "            if 'SKU' not in df.columns:\n",
    "                raise AttributeError(\"the given dataframe should contain the sku column\")\n",
    "        except Exception as other_exception:\n",
    "            print(other_exception)       \n",
    "            \n",
    "        self.target = target\n",
    "        cols = df.select_dtypes(exclude=[\"number\",\"bool_\"]).columns.to_list()\n",
    "        lb_make = LabelEncoder()\n",
    "        df_enc = df.copy()\n",
    "        cols.remove(target)\n",
    "        for categorical_column in cols:\n",
    "            df_enc[categorical_column] =  lb_make.fit_transform(df[categorical_column])  \n",
    "        \n",
    "        cols = df_enc.loc[:, df_enc.columns != target].columns\n",
    "        scaler = MinMaxScaler()\n",
    "        df_enc[cols]  = scaler.fit_transform(df_enc[cols])\n",
    "        \n",
    "        self.df = df\n",
    "        self.df_enc = df_enc\n",
    "             \n",
    "    def matrix(self):\n",
    "        sku_list = self.df_enc[self.target]\n",
    "        similarity_df = squareform(pdist(self.df_enc.loc[:, self.df_enc.columns != self.target], metric='cosine'))\n",
    "        similarity_df = pd.DataFrame(similarity_df)\n",
    "        similarity_df = 1-similarity_df\n",
    "        similarity_df = similarity_df.set_index(sku_list)\n",
    "        similarity_df.columns = sku_list\n",
    "        \n",
    "        self.similarity_df = similarity_df\n",
    "        self.sku_list = sku_list\n",
    "        return self.similarity_df.head()\n",
    "\n",
    "    def save(self, fileName, specific_path=False):\n",
    "        if specific_path:\n",
    "            self.similarity_df.to_csv(specific_path + '/' + str(fileName) + '.csv')\n",
    "        else:\n",
    "            self.similarity_df.to_csv(str(fileName) +'.csv')\n",
    "        \n",
    "    def load(self, fileName, from_path=False):\n",
    "        if from_path:\n",
    "            self.similarity_df = pd.read_csv(from_path + '/' + fileName + '.csv')\n",
    "        else:\n",
    "            self.similarity_df = pd.read_csv(fileName + '.csv')\n",
    "        self.similarity_df = self.similarity_df.set_index(self.sku_list)\n",
    "        return self.similarity_df.head()\n",
    "      \n",
    "    def recommendation(self, testSKU, number_of_similar):\n",
    "        final = self.similarity_df[[testSKU]]\n",
    "        final = final.nlargest(number_of_similar+1, testSKU)\n",
    "        final = final[final[testSKU] < 1]\n",
    "        #matches = pd.merge(final, self.df, \n",
    "         #                  on = self.target, how = \"left\").rename(columns={testSKU: \"Similarity Score\"})\n",
    "        \n",
    "        return final"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
