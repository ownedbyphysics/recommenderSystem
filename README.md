# recommenderSystem
A case study regarding Recommender System. 

## DESCRIPTION BASED:

This is an approach on text similarity technique referring to the watch industry. 
The dataset was crawled from the web and consists of the unique id (SKU) of every watch and a text description provided by the site.
For obvious reasons the SKU column has been reducted to the following form --> watch1, watch2, watch3...

** Most of online tutorials just show how to compute a pairwise similarity. But what when you need for industry reasons to keep on making queries for 
lets say the n top similar products? 
The description class firstly computes the similarity matrix for all watches, saves it to a .csv file, and then via a recommend method sends a query to 
get the answer in a dataframe result containing the SKUs of the top similar watches along with their corresponding score. See image1:





##****** The repo will be soon enriched with 2 more methods. The first one exploit the watches image similarities to get out the most similar,
and the second one produces another similarity matrix based on all watches feature scores ******##
