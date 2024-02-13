def test_pca_kmeans_clustering_runs():
    try:
        from tes import PCA
        path = './module/tests/data/Run1.1/'
        pca = PCA(path)
        pca.KMeans_clustering(4)
        executed_without_error = True
    except Exception as e:
        executed_without_error = False
    
    assert executed_without_error, "PCA.KMeans_clustering failed to execute properly."