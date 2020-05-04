import json
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.cluster import SpectralClustering
import os
import pickle

'''# TODO: There are data access oriented functions that
should be moved into their own module to clean up the code.
'''

def __load_data(data_loc):
    """Simply reads data stored in the specified
    destination and returns a python json object.

    Args:
        data_loc (str): The location of the json
            data to read.

    Returns:
        A python json object containing the data
        specified by data_loc.

    Note:
        Absolutely no deserialization is performed.
        This function does nothing more than return a
        json object.
    """
    if not data_loc.endswith('.json'):
        raise ValueError('data_loc must be a json file location.')
    with open(data_loc, 'rb') as f:
        return json.load(f)

def __title_context_dict(data):
    """Creates a dictionary of title context pairs.
    Context is the content of the wikipedia article.

    Args:
        data (dict, str): dictionary containing data,
            or json file location of squad 2.0 data.

    Returns:
        title-context dictionary
    """
    d = {}
    if isinstance(data, str):
        raw_data = __load_data(data)['data']
    else:
        raw_data = data['data']
    for i in range(len(raw_data)):
        title = raw_data[i]['title']
        paragraphs = raw_data[i]['paragraphs']
        text = []
        for j in range(len(paragraphs)):
            text.append(paragraphs[j]['context'])
        context = ' '.join(text)
        d[title] = context
    return d

def __get_data_with_titles(data, titles):
    if isinstance(data, str):
        raw_data = __load_data(data)['data']
    else:
        raw_data = data['data']
    d = {}
    d['version'] = 'v2.0'
    d['data'] = []
    for title in titles:
        for obj in raw_data:
            if obj['title'] == title:
                d['data'].append(obj)
                break
    return d

def __title_context_df(data):
    """Creates pandas DataFrame with the columns
    'title' and 'context'

    Args:
        data (dict, str): dictionary containing data,
            or json file location of squad 2.0 data.

    Returns:
        DataFrame with columns mentioned above.
    """
    if isinstance(data, str):
        data = __load_data(data)
    data_dict = __title_context_dict(data)
    d = {'title':list(data_dict.keys()), 'context':list(data_dict.values())}
    return pd.DataFrame(data=d)

def __title_cluster_df(data, num_clusters, vectorizer):
    """Clusters data and creates a dataframe containing titles and
    corresponding cluster labels. Labels will be integers starting
    at 0.

    Args:
        data (dict, str): dictionary containing data,
            or json file location of squad 2.0 data.
        num_clusters (int): The number of clusters to use.
        vectorizer (sklearn TfidfVectorizer): a vectorizer instance
            which has been trained on the data found in data_loc.
    Returns:
        DataFrame with article titles and corresponding cluster
        labels.
    """
    df = __title_context_df(data)
    df['vecs'] = vectorizer.transform(df['context']).todense().tolist()
    clusters = SpectralClustering(num_clusters).fit_predict(df['vecs'].to_list())
    df['cluster'] = clusters
    return df[['cluster', 'title']]

def get_contexts(data):
    """Returns a list of the contexts for the supplied
    data.

    Args:
        data (dict, str): dictionary containing data,
            or json file location of squad 2.0 data.
    Returns:
        list of strings containing data set contexts.
    """
    df = __title_context_df(data)
    return df['context'].to_list()

def cluster_data(data_loc, num_clusters, base_destination, vectorizer):
    """Clusters data and saves each cluster as a json file. files
    are saved in base_destination with names 'cluster_1.json',
    'cluster_2.json'... cluster statistics are stored in the
    directory specified by base_destination. The vectorizer
    supplied is also saved in base_destination as 'vectorizer.pkl'

    Args:
        data_loc (str): json file location containing squad 2.0
            data.
        num_clusters (int): The number of clusters to produce.
        base_destination (str): base folder location to save
            clustered json files to.
        vectorizer (sklearn TfidfVectorizer): a vectorizer instance
            which has been trained on the data found in data_loc.
    """
    cluster_df = __title_cluster_df(data_loc, num_clusters, vectorizer)
    if not os.path.isdir(base_destination):
        os.mkdir(base_destination)
    vec_path = os.path.join(base_destination, 'vectorizer.pkl')
    with open(vec_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    cluster_stats = {}
    for i in range(num_clusters):
        titles = cluster_df[cluster_df['cluster']==i]['title']
        cluster_stats[i] = titles.shape[0]
        cluster_data = __get_data_with_titles(data_loc, titles)
        dest = os.path.join(base_destination, 'cluster_{}.json'.format(i))
        with open(dest, 'w') as f:
            json.dump(cluster_data, f)
    stats_path = os.path.join(base_destination, 'cluster_statistics.txt')
    with open(stats_path, 'w') as f:
        for cluster in cluster_stats.keys():
            f.write('cluster {}: '.format(cluster))
            f.write(str(cluster_stats[cluster]) + '\n')

def hier_spectral(data, base_destination, branch_factor=2,
        max_size=20, _curr_label=[0], _cluster_stats={}, _root=True):
    """Clusters data and saves each cluster as a json file. files
    are saved in base_destination with names 'cluster_1.json',
    'cluster_2.json'... cluster statistics are stored in the
    directory specified by base_destination. The vectorizer
    supplied is also saved in base_destination as 'vectorizer.pkl'

    Clustering is done via a hierarchical method. Rather than supply
    the number of total clusters as input, you supply a branch factor,
    and max cluster size to control the size of clusters. The algorithm
    consists of recursive stages of clustering. At each stage, if a cluster
    is larger than the max cluster size provided, the algorithm is recursively
    called. The terminating base case is when all clusters are <= max
    cluster size.

    Args:
        data (dict, str): dictionary containing data,
            or json file location of squad 2.0 data.
        base_destination (str): base folder location to save
            clustered json files to.
        branch_factor (int, default 2): the number of clusters to
            produce at each recursive step of the algorithm.
        max_size (int, default 20): The max size a cluster can
            be.
    """
    vec = TfidfVectorizer().fit(get_contexts(data))
    cluster_df = __title_cluster_df(data, branch_factor, vec)
    if not os.path.isdir(base_destination):
        os.mkdir(base_destination)
    if _root:
        vec_path = os.path.join(base_destination, 'vectorizer.pkl')
        with open(vec_path, 'wb') as f:
            pickle.dump(vec, f)
    for i in range(branch_factor):
        titles = cluster_df[cluster_df['cluster']==i]['title']
        cluster_size = titles.shape[0]
        cluster_data = __get_data_with_titles(data, titles)
        if cluster_size <= max_size:
            dest = os.path.join(base_destination, 'cluster_{}.json'.format(_curr_label[0]))
            with open(dest, 'w') as f:
                json.dump(cluster_data, f)
            _cluster_stats[_curr_label[0]] = cluster_size
            _curr_label[0] += 1
        else:
            hier_spectral(cluster_data, base_destination, branch_factor,
                _curr_label, _cluster_stats, _root=False)
    if _root:
        stats_path = os.path.join(base_destination, 'cluster_statistics.txt')
        with open(stats_path, 'w') as f:
            for cluster in _cluster_stats.keys():
                f.write('cluster {}: '.format(cluster))
                f.write(str(_cluster_stats[cluster]) + '\n')
