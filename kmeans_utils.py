import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from kneed import KneeLocator
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from statistics import mode
from pandas.api.types import is_numeric_dtype


class MyKM:
    def __init__(self, n_clusters, seed=42, max_iter=100, eps=0.1):
        """Init method"""
        self.seed = seed
        self.max_iter = max_iter
        self.eps = eps
        self.n_clusters = n_clusters
    
    def __str__(self):
        """str method"""
        return 'MyKM'

            
    def fit(self, X, info=False):
        """Fit method"""
        # Initialize k and X(=dataFrame)
        self.X = X
        _X = X.copy().to_numpy()
        t0 = time.time()

        # Initialize the centroids with a simple sampling
        centroids = self.X.sample(n=self.n_clusters, random_state=self.seed).to_numpy()        
        # Initialize the stop condition
        max_iter_cond = self.max_iter
        eps_cond = True
        
        # Compute the labels for all the points
        labels = np.apply_along_axis(self._compute_label, 1, _X, centroids)
        
        # Start the training
        while all(
                (max_iter_cond, eps_cond,)
                ):
            # Update the centroids
            prev_centroids = centroids.copy()
            centroids = self._update_centroids(labels)
            # Assign the new labels to the data
            labels = np.apply_along_axis(self._compute_label, 1, _X, centroids)
            # update max iter condition
            max_iter_cond -= 1
            # check the centroid update condition
            eps_cond = self._get_eps_cond(centroids, prev_centroids)
          
        # Get info about stop condition: True means that the condition
        # stopped the loop.
        self.stop_condition = {
                          # "switch": switch_cond == False, 
                          "max_iter": max_iter_cond == False, 
                          "eps": eps_cond == False
                         }
        self.labels_ = labels
        self.centroids = centroids
        # Calculate the inertia (i.e: the total distance between data-points 
        # and their own centroid) 
        self._get_inertia()
        self.elapsed_time = time.time() - t0
        # Get useful informations about the training
        if info:
            self.get_info()
            
            
    def _compute_label(self, row, centroids_):
        """Assign labels to the data"""
        # Calculate the Euclideian distance between data and all the centroids
        # dists = centroids_.apply(lambda x: np.linalg.norm(x - row), axis=1)
        dists = np.apply_along_axis(self._compute_norm, 1, centroids_, row)
        
        # Return the label with min distance
        return np.argmin(dists)
        
    def _compute_norm(self, x1, x2):
        """Compute the euclideian distance between two vectors"""
        return np.linalg.norm(x1 - x2)
                
    def _update_centroids(self, labels_):
        """Update the centroids' values"""
        X_temp = self.X.copy()
        X_temp['labels'] = labels_
        
        # For each group(i.e.: label value), calculate the mean value. It
        # returns a dataframe with values for each centroid
        return X_temp.groupby('labels').mean().reset_index(drop=True)
    
    
    def _get_inertia(self):
        """Calculate the inertia of the clustering"""
        X_ = self.X.copy()
        X_['labels'] = self.labels_
        # Calculate the distance for each data-point to its own center and
        # then sum all the distances
        self.inertia_ = X_.apply(self._get_dist_from_centroid, axis=1).sum()
        
        
    def _get_dist_from_centroid(self, row):
        """Calculate the distance between a sample and its own centroid"""
        # Unpack the row (i.e.: pandas.Series)
        row_, label_ = row[:-1], int(row.iat[-1])
        # Get the centroid for a specific label
        centroid_ = self.centroids.iloc[label_, :]
        
        # Return the Euclideian distance
        return np.linalg.norm(row_ - centroid_)
    
    def _get_eps_cond(self, new_centroids, old_centroids):
        """Calculate the distance between the new centroids and the ones
           at the previous step."""
        return np.linalg.norm(
                        new_centroids - old_centroids, axis=1
                        )\
                        .sum() > self.eps * self.X.shape[1]
        
    def get_info(self):
        try:
            print(
                "\n*** INFO: ***",
                f"n_clusters: {self.n_clusters}",
                f"Elapsed time: {self.elapsed_time}",
                f"Stop Condtion: {[name for name, value in self.stop_condition.items() if value]}",
                f"Inertia: {self.inertia_}",
                sep='\n'
                )
        except AttributeError:
            print('Run fit before getting info!')

    
def run_knee_method(max_k, **kwargs):
    inertia_scores = kwargs.get('inertia_scores')
    range_ = range(1, max_k + 1)
      
    knee = KneeLocator(
       range_, inertia_scores, curve="convex", direction="decreasing"
       )
    
    if knee.knee is not None:
        knee.plot_knee()
        return knee.knee  
    return -1
        

def run_silhouette_method(max_k, **kwargs):
    """Compute best k with silhouette method"""
    silhouette_scores = kwargs.get('silhouette_scores')
    # Get max value of silhouette_scores (i.e.: best score)
    max_value = max(silhouette_scores)
    # 2 + list index (i.e.: from 0 to n). silhouette_scores can be computed
    # with 2 or more centroids, so our list starts from the score of k=2
    best_k = 2 + silhouette_scores.index(max_value)
    # Plot the silhouette scores
    fig, ax = plt.subplots()
    ax.plot(range(2, max_k + 1), silhouette_scores)
    ax.vlines(x=best_k, ymin=0, ymax=max_value, linestyles='--', color='cyan')
    plt.xlabel("# of Clusters")
    plt.ylabel("Silhouette Score")
    plt.show()
    
    return best_k


def run_calinski_harabasz_method(max_k, **kwargs):
    """Compute best k with silhouette method"""
    calinski_harabasz_scores = kwargs.get('calinski_harabasz_scores')
    # Get max value of calinski_harabasz_scores (i.e.: best score)
    max_value = max(calinski_harabasz_scores)
    # 2 + list index (i.e.: from 0 to n). silhouette_scores can be computed
    # with 2 or more centroids, so our list starts from the score of k=2
    best_k = 2 + calinski_harabasz_scores.index(max_value)
    # Plot the Calinski Harabasz scores
    fig, ax = plt.subplots()
    ax.plot(range(2, max_k + 1), calinski_harabasz_scores)
    ax.vlines(x=best_k, ymin=0, ymax=max_value, linestyles='--', color='cyan')
    plt.xlabel("# of Clusters")
    plt.ylabel("calinski Harabasz Score")
    plt.show()
    
    return best_k


# Dictionay with the available methods for computing the best k
eval_methods_dict = {
    'knee': run_knee_method,
    'silhouette': run_silhouette_method,
    'calinski_harabasz': run_calinski_harabasz_method
    }       


def get_best_k(
        model, X, max_k=20, eval_methods=['knee', 'silhouette', 'calinski_harabasz'], **kwargs
               ):
    """Compute best k given several methods"""
    results = {}
    inertia_scores = []
    silhouette_scores = []
    calinski_harabasz_scores = []
    
    for k in range(1, max_k + 1):
        try:
            model_ = model(n_clusters=k, **kwargs)
        except AttributeError as ex:
            print('Warning! -', ex)
            model_ = model(n_clusters=k)
        print(f'Fitting: {str(model_)} with k={k}')
        # Fit the model for a given k
        model_.fit(X)
        # Retrieve the scores
        inertia_scores.append(model_.inertia_)
        if k > 1:
            silhouette_scores.append(
                silhouette_score(X, model_.labels_)
                )
            calinski_harabasz_scores.append(
                calinski_harabasz_score(X, model_.labels_)
                )
                
    # Compute the scores with the different methods
    for name in eval_methods:
        if name in eval_methods_dict.keys():
            results[name] = eval_methods_dict[name](
                                max_k,
                                inertia_scores=inertia_scores, 
                                silhouette_scores=silhouette_scores, 
                                calinski_harabasz_scores=calinski_harabasz_scores,
                                **kwargs
                                )
        else:
            # If a wrong method is given
            NotImplementedError('Method still not supported')
    
    k_list = list(results.values())
    # Choose best k as mode of the scores. If no unique mode is found, the 
    #first element of the mode will be provided( by default: knee_method)
    best_k = mode(k_list)
    
    return best_k, results


def save_model(model, name='model'):
    """Saving the model as pickle file"""
    with open(name + '.pkl', 'wb') as file:
        pickle.dump(model, file)
        
        
def load_model(name='model'):
    """Load the model from a pickle file"""
    with open(f'{name}.pkl', 'rb') as file:
        model = pickle.load(file)
        
    return model
    

def get_crosstab(name, df):
    """Compute the crosstab normalized by column with marginal total"""
    df_ = df.copy()
    # If the column is numeric, convert it into 4 bins
    if is_numeric_dtype(df[name]):
        df_[name] =  pd.qcut(df_[name], q=4)
    # Get labels values and sort
    labels = df_['labels'].unique()
    labels.sort()
    # Get the cross table, normalized by columns
    cross_table = pd.crosstab(df_[name], df_['labels'], normalize='columns') * 100
    # Get the colums total, as check
    total = pd.Series(cross_table.sum(axis=0).to_list(), name='Total')
    # Add the total to the cross_table and return it
    cross_table = round(cross_table.append(total), 1)
    cross_table.columns = [f'Cluster # {i + 1}' for i in labels]
    return cross_table