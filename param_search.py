from hyperopt import fmin, partial, space_eval, tpe, Trials, STATUS_OK
from clustering import generate_clusters, score_clusters


'''
Objective function to be minimised in Bayesian hyperparameter search
====================================================================
Parameters:
    - params: dictionary of parameters for generate_clusters function (UMAP and HDBSCAN)
    - embeddings: np.array-like structure of embeddings for dimension reduction and clustering
    - label_lower: no. of clusters must not be less than this value
    - label_upper: no. of clusters must not be more than this value 
    - penalty_lvl: penalty imposed if no. of clusters fall out of range of specified bounds 
'''


def objective(params, embeddings, label_lower, label_upper, penalty_lvl=0.15):
    # obtain HDBSCAN clusters from embeddings
    clusters, _ = generate_clusters(embeddings,
                                 n_neighbors=params['n_neighbors'],
                                 n_components=params['n_components'],
                                 min_cluster_size=params['min_cluster_size'],
                                 random_state=params['random_state'])

    # obtain number of clusters and cost associated with the cluster
    label_count, cost = score_clusters(clusters, prob_threshold=0.05)

    # impos extra penalty if no. of clusters is outside the specified range
    if (label_count < label_lower) | (label_count > label_upper):
        penalty = penalty_lvl
    else:
        penalty = 0

    loss = cost + penalty

    return {'loss': loss, 'label_count': label_count, 'status': STATUS_OK}


'''
Perform Bayesian hyperparameter optimisation of minimise objective function
===========================================================================
Parameters:
    - embeddings: np.array-like structure of embeddings for dimension reduction and clustering
    - space: dictionary of hyperparameter choices, e.g.
        {'n_neighbors': hp.choice('n_neighbors', range(5,20)), ...}
    - label_lower: no. of clusters must not be less than this value
    - label_upper: no. of clusters must not be more than this value
'''


def bayesian_search(embeddings, space, label_lower, label_upper, max_evals=100):
    trials = Trials()
    fmin_objective = partial(objective, embeddings=embeddings, label_lower=label_lower, label_upper=label_upper)
    best = fmin(fmin_objective,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials)

    best_params = space_eval(space, best)
    print('best:')
    print(best_params)
    print(f"label count: {trials.best_trial['result']['label_count']}")

    # generate clusters with the best hyperparameters found
    best_clusters, _ = generate_clusters(embeddings,
                                      n_neighbors=best_params['n_neighbors'],
                                      n_components=best_params['n_components'],
                                      min_cluster_size=best_params['min_cluster_size'],
                                      random_state=best_params['random_state'])

    return best_params, best_clusters, trials