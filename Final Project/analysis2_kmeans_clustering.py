# K-means clustering

print('Lets start!')

# Packages
import pandas as pd
import re
import sklearn
import sklearn.feature_extraction.text
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.datasets
import sklearn.cluster
import sklearn.decomposition
import sklearn.metrics
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm
import seaborn as sns 

## Helper functions
import util_analysis1_word_count as util

def tf_idfVectorizer(df, input_text, stp_wrds=[]):
    '''
    Creates a TfidfVectorizer class using skit learn and fits it to an input text
    from a pandas df
    
    Input:
        df(pandas df): dataframe
        input_text(pandas serias): pandas columns with tokenized text
        stp_wrds(list): list of stop words
    
    Output:
        dfTFVectorizer(class): class that sarisfy conditions within function 
        dfTFVects(fit model): matrix of vectorized text usinf tf idf
    '''
    dfTFVectorizer = \
            sklearn.feature_extraction.text.TfidfVectorizer(max_df=.5, 
                                                            max_features=1000, 
                                                            min_df=3, 
                                                            norm='l2',
                                                            stop_words=stp_wrds)
    dfTFVects = dfTFVectorizer.fit_transform(df[input_text])
    
    return dfTFVectorizer, dfTFVects
    
def silhouette(n_clusters, X):
    '''
    Returns the average silhouette score
    
    Input: 
        n_clusters(int): number of clusters
        X(np array): array representation of the wordVect matrix

    Output:
        silhouette_avg(float): average silhouette score
    '''
    clusterer = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = sklearn.metrics.silhouette_score(X, cluster_labels)
    
    return silhouette_avg


def list_optimal(list_tfvects_arrays, max_nclust, min_nclust=2):
    '''
    Returns a list with the optimal number of clusters and 
    the average silhouette score for that optimal number
    
    Input: 
        list_tfvects_arrays(list): list of mujicaTFVects as numpy array
        min_nclust(int): min number of clusters 
        max_nclust(int): max number of clusters
    
    Output:
        list_optimal(list): list of tuples
                            tuple[0]: optimal_silhouette_score
                            tuple[1]: optimal_nclust
    '''
    list_optimal = []
    for X in list_tfvects_arrays:
        optimal_silhouette_score = 0
        for n_clusters in range(min_nclust,max_nclust):
            silhouette_score = silhouette(n_clusters, X)
            if optimal_silhouette_score < silhouette_score:
                optimal_silhouette_score = silhouette_score
                optimal_nclust = n_clusters
        list_optimal.append((optimal_silhouette_score, optimal_nclust))
    
    return list_optimal

print('functions working! I think')

## Data
data = pd.read_csv('data/2021_03_data.csv')
corpus = data

## Text preprocesing:
## Delete duplicates
corpus = corpus.drop_duplicates(subset=['Text'])


# Lematize 
# corpus['lemmatized_text'] = \
#    corpus['normalized_text'].apply(lambda x: lematize_list(x))


# Create a dataset in which each politician is mentioned and 
# the source comes from the politician‘s country
santos = corpus[(corpus["Text"].str.contains("Santos")) & (corpus["Country"]=="CO")]
uribe = corpus[(corpus["Text"].str.contains("Uribe")) & (corpus["Country"]=="CO")]
pena_nieto = corpus[(corpus["Text"].str.contains("Peña Nieto")) & (corpus["Country"]=="MX")]
correa = corpus[(corpus["Text"].str.contains("Correa")) & (corpus["Country"]=="EC")]
morales = corpus[corpus["Text"].str.contains("Evo") & (corpus["Country"]=="BO")]
chavez = corpus[corpus["Text"].str.contains("Chávez") & (corpus["Country"]=="VE")]
maduro = corpus[corpus["Text"].str.contains("Maduro") & (corpus["Country"]=="VE")]
kirchner = corpus[corpus["Text"].str.contains("Kirchner") & (corpus["Country"]=="AR")]
ortega = corpus[corpus["Text"].str.contains("Ortega") & (corpus["Country"]=="NI")]
bachelet = corpus[corpus["Text"].str.contains("Bachelet") & (corpus["Country"]=="CL")]
mujica = corpus[corpus["Text"].str.contains("Mujica") & (corpus["Country"]=="UY")]
print('data frames for each president created')

list_df = [santos, uribe, pena_nieto, correa,\
            morales, chavez, maduro, kirchner, ortega, bachelet, mujica]

list_names = ['santos', 'uribe', 'pena_nieto', 'correa',\
              'morales', 'chavez', 'maduro', 'kirchner', 'ortega', 
              'bachelet', 'mujica']

i = 0
for df in list_df:
    df.name = list_names[i]
    i += 1

for df in list_df:
    print(df.name, 'df:', df.shape)

for df in list_df:
    df['text2'] = df['Text'].apply(lambda x: re.sub('[¡!@#$:).;,¿?&]', '', x.lower()))
    df['text2'] = \
            df['text2'].apply(lambda x: re.sub("\d+", "", x))


stp_wrds = ['me', 
 'mi', 
 'yo', 
 'era', 
 'había', 
 'muy', 
 'estaba',
 'qué', 
 'he', 
 'día', 
 'tnn', 
 'me',
 'qué',
 'ni', 
 'gente', #I don't think you want to take this word out. 
 'muy', 
 'yo', 
 'bien', #I don't think you want to take this word out.
 'decir',  
 'puede', 
 'esa', 
 'te', 
 'usted']

santosTFVectorizer, santosTFVects = tf_idfVectorizer(santos, 'text2', stp_wrds)
uribeTFVectorizer, uribeTFVects = tf_idfVectorizer(uribe, 'text2', stp_wrds)
pena_nietoTFVectorizer, pena_nietoTFVects = tf_idfVectorizer(pena_nieto, 'text2', stp_wrds)
correaTFVectorizer, correaTFVects = tf_idfVectorizer(correa, 'text2', stp_wrds)
moralesTFVectorizer, moralesTFVects = tf_idfVectorizer(morales, 'text2', stp_wrds)
chavezTFVectorizer, chavezTFVects = tf_idfVectorizer(chavez, 'text2', stp_wrds)
maduroTFVectorizer, maduroTFVects = tf_idfVectorizer(maduro, 'text2', stp_wrds)
kirchnerTFVectorizer, kirchnerTFVects = tf_idfVectorizer(kirchner, 'text2', stp_wrds)
ortegaTFVectorizer, ortegaTFVects = tf_idfVectorizer(ortega, 'text2', stp_wrds)
bacheletTFVectorizer, bacheletTFVects = tf_idfVectorizer(bachelet, 'text2', stp_wrds)
mujicaTFVectorizer, mujicaTFVects = tf_idfVectorizer(mujica, 'text2', stp_wrds)

# List of TFVectors
list_tfvects = [santosTFVects, uribeTFVects, pena_nietoTFVects,
                correaTFVects, moralesTFVects, chavezTFVects,
                maduroTFVects, kirchnerTFVects, ortegaTFVects, 
                bacheletTFVects, mujicaTFVects] 

print('list_tfvects created')
'''
Silhouette Scores
                
From Wikipedia, the free encyclopedia

Silhouette refers to a method of interpretation and validation of consistency within clusters of data. 
The technique provides a succinct graphical representation of how well each object has been classified.[1]

The silhouette value is a measure of how similar an object is to its own cluster (cohesion) 
compared to other clusters (separation). The silhouette ranges from −1 to +1, where a high 
value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters. 
If most objects have a high value, then the clustering configuration is appropriate. 
If many points have a low or negative value, then the clustering configuration may have too many or too few clusters.

The silhouette can be calculated with any distance metric, such as the Euclidean distance or the Manhattan distance. 
'''
print('starting sil score analysis:')            
'''
list_tfvects_arrays = []
for tfvect in list_tfvects:
    X = tfvect.toarray()
    list_tfvects_arrays.append(X)

list50_clusters = list_optimal(list_tfvects_arrays, 50, 2)    
list100_clusters = list_optimal(list_tfvects_arrays, 100, 2)

df_silhscores50 = pd.DataFrame(list(zip(list_names, list50_clusters)), 
                       columns =['Name', 'val']) 

df_silhscores100 = pd.DataFrame(list(zip(list_names, list100_clusters)), 
                       columns =['Name', 'val']) 

df_silhscores50.to_csv('results/kmeans_clustering/silhouette_scores50clusters.csv')  
df_silhscores100.to_csv('results/kmeans_clustering/silhouette_scores100clusters.csv')
''' 

        

################################################################################
################################################################################
'''
What's really interesting below are the low avg silh scores for 
the left=wing populist candidates, which suggests more dispersion
around them

I did an in-depth evaluation of the clusters and found them to be 
consistent

          Name                         val
0       santos   (0.04750108542063177, 46)
1        uribe   (0.07287269766037982, 49)
2   pena_nieto   (0.05435532376132274, 33)

3       correa  (0.034512983488498884, 49)
4      morales  (0.026150902623854385, 43)
5       chavez  (0.027139594907412776, 49)
6       maduro   (0.04196562809742811, 44)
7     kirchner    (0.0423301681051237, 43)
8       ortega  (0.045807589930400024, 19)

9     bachelet   (0.05051754683771609, 49)
10      mujica    (0.0421273085928835, 45) 

           Name                         val
0       santos   (0.05255015971214607, 84)
1        uribe   (0.07331839555375061, 81)
2   pena_nieto   (0.06763719717578481, 90)

3       correa  (0.038705626475398346, 99)
4      morales   (0.03225701030028797, 94)
5       chavez   (0.03322364532018523, 97)
6       maduro   (0.05151521753702726, 93)
7     kirchner   (0.04805310664441619, 98)
8       ortega  (0.047774160904365814, 78)

9     bachelet   (0.06083706272511013, 99)
10      mujica   (0.04576991732029961, 98)


################################################################################
################################################################################
'''
df_silhscores50 = pd.read_csv('results/kmeans_clustering/silhouette_scores50clusters.csv')   
df_silhscores100 = pd.read_csv('results/kmeans_clustering/silhouette_scores100clusters.csv')   


numCategories = 46
santosKM = sklearn.cluster.KMeans(n_clusters = numCategories, init='k-means++')
santosKM.fit(santosTFVects)
santosPCA = sklearn.decomposition.PCA(n_components = 2).fit(santosTFVects.toarray())
santosPCA_data = santosPCA.transform(santosTFVects.toarray())
colors = list(plt.cm.rainbow(np.linspace(0,1, numCategories)))
colors_p = [colors[l] for l in santosKM.labels_]
fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.set_frame_on(False)
plt.scatter(santosPCA_data[:, 0], santosPCA_data[:, 1], color = colors_p, alpha = 0.5)
plt.xticks(())
plt.yticks(())
plt.title('Predicted Clusters\n k = {}'.format(numCategories))
plt.savefig('results/kmeans_clustering/figures50clusters/santos50.pdf')


numCategories = 49
uribeKM = sklearn.cluster.KMeans(n_clusters = numCategories, init='k-means++')
uribeKM.fit(uribeTFVects)
uribePCA = sklearn.decomposition.PCA(n_components = 2).fit(uribeTFVects.toarray())
uribePCA_data = uribePCA.transform(uribeTFVects.toarray())
colors = list(plt.cm.rainbow(np.linspace(0,1, numCategories)))
colors_p = [colors[l] for l in uribeKM.labels_]
fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.set_frame_on(False)
plt.scatter(uribePCA_data[:, 0], uribePCA_data[:, 1], color = colors_p, alpha = 0.5)
plt.xticks(())
plt.yticks(())
plt.title('Predicted Clusters\n k = {}'.format(numCategories))
plt.savefig('results/kmeans_clustering/figures50clusters/uribe50.pdf')


numCategories = 33
pena_nietoKM = sklearn.cluster.KMeans(n_clusters = numCategories, init='k-means++')
pena_nietoKM.fit(pena_nietoTFVects)
pena_nietoPCA = sklearn.decomposition.PCA(n_components = 2).fit(pena_nietoTFVects.toarray())
pena_nietoPCA_data = pena_nietoPCA.transform(pena_nietoTFVects.toarray())
colors = list(plt.cm.rainbow(np.linspace(0,1, numCategories)))
colors_p = [colors[l] for l in pena_nietoKM.labels_]
fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.set_frame_on(False)
plt.scatter(pena_nietoPCA_data[:, 0], pena_nietoPCA_data[:, 1], color = colors_p, alpha = 0.5)
plt.xticks(())
plt.yticks(())
plt.title('Predicted Clusters\n k = {}'.format(numCategories))
plt.savefig('results/kmeans_clustering/figures50clusters/pena_nieto50.pdf')


numCategories = 49
correaKM = sklearn.cluster.KMeans(n_clusters = numCategories, init='k-means++')
correaKM.fit(correaTFVects)
correaPCA = sklearn.decomposition.PCA(n_components = 2).fit(correaTFVects.toarray())
correaPCA_data = correaPCA.transform(correaTFVects.toarray())
colors = list(plt.cm.rainbow(np.linspace(0,1, numCategories)))
colors_p = [colors[l] for l in correaKM.labels_]
fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.set_frame_on(False)
plt.scatter(correaPCA_data[:, 0], correaPCA_data[:, 1], color = colors_p, alpha = 0.5)
plt.xticks(())
plt.yticks(())
plt.title('Predicted Clusters\n k = {}'.format(numCategories))
plt.savefig('results/kmeans_clustering/figures50clusters/correa50.pdf')


numCategories = 43
moralesKM = sklearn.cluster.KMeans(n_clusters = numCategories, init='k-means++')
moralesKM.fit(moralesTFVects)
moralesPCA = sklearn.decomposition.PCA(n_components = 2).fit(moralesTFVects.toarray())
moralesPCA_data = moralesPCA.transform(moralesTFVects.toarray())
colors = list(plt.cm.rainbow(np.linspace(0,1, numCategories)))
colors_p = [colors[l] for l in moralesKM.labels_]
fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.set_frame_on(False)
plt.scatter(moralesPCA_data[:, 0], moralesPCA_data[:, 1], color = colors_p, alpha = 0.5)
plt.xticks(())
plt.yticks(())
plt.title('Predicted Clusters\n k = {}'.format(numCategories))
plt.savefig('results/kmeans_clustering/figures50clusters/morales50.pdf')


numCategories = 49
chavezKM = sklearn.cluster.KMeans(n_clusters = numCategories, init='k-means++')
chavezKM.fit(chavezTFVects)
chavezPCA = sklearn.decomposition.PCA(n_components = 2).fit(chavezTFVects.toarray())
chavezPCA_data = chavezPCA.transform(chavezTFVects.toarray())
colors = list(plt.cm.rainbow(np.linspace(0,1, numCategories)))
colors_p = [colors[l] for l in chavezKM.labels_]
fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.set_frame_on(False)
plt.scatter(chavezPCA_data[:, 0], chavezPCA_data[:, 1], color = colors_p, alpha = 0.5)
plt.xticks(())
plt.yticks(())
plt.title('Predicted Clusters\n k = {}'.format(numCategories))
plt.savefig('results/kmeans_clustering/figures50clusters/chavez50.pdf')


numCategories = 44
maduroKM = sklearn.cluster.KMeans(n_clusters = numCategories, init='k-means++')
maduroKM.fit(maduroTFVects)
maduroPCA = sklearn.decomposition.PCA(n_components = 2).fit(maduroTFVects.toarray())
maduroPCA_data = maduroPCA.transform(maduroTFVects.toarray())
colors = list(plt.cm.rainbow(np.linspace(0,1, numCategories)))
colors_p = [colors[l] for l in maduroKM.labels_]
fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.set_frame_on(False)
plt.scatter(maduroPCA_data[:, 0], maduroPCA_data[:, 1], color = colors_p, alpha = 0.5)
plt.xticks(())
plt.yticks(())
plt.title('Predicted Clusters\n k = {}'.format(numCategories))
plt.savefig('results/kmeans_clustering/figures50clusters/maduro50.pdf')

numCategories = 43
kirchnerKM = sklearn.cluster.KMeans(n_clusters = numCategories, init='k-means++')
kirchnerKM.fit(kirchnerTFVects)
kirchnerPCA = sklearn.decomposition.PCA(n_components = 2).fit(kirchnerTFVects.toarray())
kirchnerPCA_data = kirchnerPCA.transform(kirchnerTFVects.toarray())
colors = list(plt.cm.rainbow(np.linspace(0,1, numCategories)))
colors_p = [colors[l] for l in kirchnerKM.labels_]
fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.set_frame_on(False)
plt.scatter(kirchnerPCA_data[:, 0], kirchnerPCA_data[:, 1], color = colors_p, alpha = 0.5)
plt.xticks(())
plt.yticks(())
plt.title('Predicted Clusters\n k = {}'.format(numCategories))
plt.savefig('results/kmeans_clustering/figures50clusters/kirchner50.pdf')


numCategories = 19
ortegaKM = sklearn.cluster.KMeans(n_clusters = numCategories, init='k-means++')
ortegaKM.fit(ortegaTFVects)
ortegaPCA = sklearn.decomposition.PCA(n_components = 2).fit(ortegaTFVects.toarray())
ortegaPCA_data = ortegaPCA.transform(ortegaTFVects.toarray())
colors = list(plt.cm.rainbow(np.linspace(0,1, numCategories)))
colors_p = [colors[l] for l in ortegaKM.labels_]
fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.set_frame_on(False)
plt.scatter(ortegaPCA_data[:, 0], ortegaPCA_data[:, 1], color = colors_p, alpha = 0.5)
plt.xticks(())
plt.yticks(())
plt.title('Predicted Clusters\n k = {}'.format(numCategories))
plt.savefig('results/kmeans_clustering/figures50clusters/ortega50.pdf')


numCategories = 49
bacheletKM = sklearn.cluster.KMeans(n_clusters = numCategories, init='k-means++')
bacheletKM.fit(bacheletTFVects)
bacheletPCA = sklearn.decomposition.PCA(n_components = 2).fit(bacheletTFVects.toarray())
bacheletPCA_data = bacheletPCA.transform(bacheletTFVects.toarray())
colors = list(plt.cm.rainbow(np.linspace(0,1, numCategories)))
colors_p = [colors[l] for l in bacheletKM.labels_]
fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.set_frame_on(False)
plt.scatter(bacheletPCA_data[:, 0], bacheletPCA_data[:, 1], color = colors_p, alpha = 0.5)
plt.xticks(())
plt.yticks(())
plt.title('Predicted Clusters\n k = {}'.format(numCategories))
plt.savefig('results/kmeans_clustering/figures50clusters/bachelet50.pdf')


numCategories = 45
mujicaKM = sklearn.cluster.KMeans(n_clusters = numCategories, init='k-means++')
mujicaKM.fit(mujicaTFVects)
mujicaPCA = sklearn.decomposition.PCA(n_components = 2).fit(mujicaTFVects.toarray())
mujicaPCA_data = mujicaPCA.transform(mujicaTFVects.toarray())
colors = list(plt.cm.rainbow(np.linspace(0,1, numCategories)))
colors_p = [colors[l] for l in mujicaKM.labels_]
fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.set_frame_on(False)
plt.scatter(mujicaPCA_data[:, 0], mujicaPCA_data[:, 1], color = colors_p, alpha = 0.5)
plt.xticks(())
plt.yticks(())
plt.title('Predicted Clusters\n k = {}'.format(numCategories))
plt.savefig('results/kmeans_clustering/figures50clusters/mujica50.pdf')


################################################################################
################################################################################

