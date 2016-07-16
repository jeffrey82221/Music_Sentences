
#REVIEW functions for NB classifier
from util_package import *

def filter_words(voc_count, fc_h, fc_l):
    import collections
    N = len(voc_count)
    voc_count.subtract(collections.Counter(
        dict(voc_count.most_common(int(N * fc_h)))))
    voc_count = collections.Counter(
        dict(voc_count.most_common(int(N * (1. - fc_h - fc_l)))))
    return collections.Counter({k: v for k, v in voc_count.items() if v > 0})
def vectorize_docs(token_list, V):
    import numpy as np
    V_dict = dict(zip(V, range(len(V))))  # use dictionary for faster indexing
    from scipy.sparse import lil_matrix
    from scipy.sparse import csr_matrix
    doc_vec_M = lil_matrix((len(V) + 1, len(token_list)), dtype=np.int16)

    def find_index(key):
        try:
            return V_dict[key]
        except:
            return -1
    find_index_map = np.vectorize(find_index)
    key_list = []
    doc_index_list = []
    count_list = []
    for i in range(len(token_list)):
        try:
            key, count = np.unique(token_list[i], return_counts=True)
            # remove keys that are not contrain in voc list
            key_list.extend(find_index_map(key).tolist())
            count_list.extend(count)
            doc_index_list.extend([i] * len(key))
        except:
            None  # print('doc ', i, ' ,: term num = ', len(key))
    doc_vec_M[key_list, doc_index_list] = np.matrix(count_list)
    return csr_matrix(doc_vec_M)[:-1, :]


def maximum_likelihood_estimation(doc_vecs, posterior, smooth_const):
    import numpy as np
    prior = np.sum(posterior, 0)
    prior = prior + smooth_const
    prior = prior / np.sum(prior)
    condprob = doc_vecs * posterior
    condprob = condprob + smooth_const
    condprob = condprob / np.sum(condprob, 0)
    return prior, condprob


# tokenize texts

def common_words(c,k,V,condprob):
    import collections
    return collections.Counter(dict(zip(V,condprob[:,c].transpose().tolist()[0]))).most_common(k)

def analysis(pd,title_data,cat_data,text_data,hfc=0.,lfc=0.):
    import collections
    import numpy as np
    # input : document classes array, documents texts
    # calculate conditional probability of each term in given a class
    # return
    # 1. the converiance matrix of each class calculated by their content
    # 2. the converiance matrix of each document calculated by their content
    # 3. the conditional probability of each term given a class
    # 4. list of the vocaburary V
    # 5. list of the classes
    # 6. prior of each classes
    # 7. show the first 10 common words for each class

    #convert pandas to numpy
    site_names = [e[0] for e in title_data.values.tolist()]
    cat_names = [e[0] for e in cat_data.values.tolist()]
    site_body = [e[0] for e in text_data.values.tolist()]

    stop_words = load_stop_words()
    site_body_tokens = [tokenize(sb,stop_words) for sb in site_body]

    vocs = filter_words(collections.Counter(merge_lists(site_body_tokens)),hfc,lfc)
    V = list(vocs)

    body_vecs_M = vectorize_docs(site_body_tokens,V)

    # setup the posterior
    cat_list = list(set(cat_names))

    posterior_train = []
    for cat in cat_names:
        class_posterior = [0.] * len(cat_list)
        class_posterior[cat_list.index(cat)] = 1.
        posterior_train.append(class_posterior)
    posterior_train_M = np.matrix(posterior_train)

    prior, condprob = maximum_likelihood_estimation(body_vecs_M, posterior_train_M, 0.1)

    # generate weighted document vectors
    weights = np.sum(condprob,1)

    weighted_doc_vec_M=body_vecs_M.toarray()*np.array(weights.tolist())


    #for i in range(len(cat_list)):
    #    print(cat_list[i])
    #    print(common_words(i,50,V,condprob))

    class_cov_M = np.cov(condprob.transpose())
    doc_cov_M = np.cov(body_vecs_M.toarray().transpose())
    prior_table = pd.DataFrame(prior,columns=cat_list,index=['prior'])
    condi_table = pd.DataFrame(condprob,columns=cat_list,index=V)
    doc_cov_table = pd.DataFrame(doc_cov_M,columns=site_names,index=site_names)
    class_cov_table = pd.DataFrame(class_cov_M,columns=cat_list,index=cat_list)
    weighted_doc_vec_table = pd.DataFrame(np.matrix(body_vecs_M.toarray()),columns=site_names,index=V)
    doc_vec_table = pd.DataFrame(np.matrix(weighted_doc_vec_M),columns=site_names,index=V)
    return prior_table,condi_table,doc_cov_table,class_cov_table,doc_vec_table,weighted_doc_vec_table

def k_nearest_neighbor(table, k):
    from sklearn.neighbors import NearestNeighbors
    import pandas as pd
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(table)
    distances, indices = nbrs.kneighbors(table)
    indices
    neighbor_list = []
    for i in range(len(table)):
        neighbor = []
        count = 0
        for index in indices[i]:
            neighbor.append(table.index[index])
            neighbor.append(distances[i][count])
            count = count + 1
        neighbor_list.append(neighbor)

    return pd.DataFrame(neighbor_list, index=table.index)
