# TSNE BENCHMARK
import time
import numpy as np
from openTSNE import TSNE, affinity, TSNEEmbedding
from umap import UMAP
from rastermap.utils import bin1d, split_testtrain
from rastermap.mapping_landmark import Rastermap
from rastermap.metrics import embedding_score

def run_TSNE(U, perplexities=[30]):
    if len(perplexities) > 1:
        affinities = affinity.Multiscale(
            U,
            perplexities=perplexities,
            metric="cosine",
            n_jobs=16,
            random_state=1,
            #verbose=True
        )
        affinities = affinity.PerplexityBasedNN(
            U,
            perplexity=perplexities[0],
            metric="cosine",
            n_jobs=16,
            random_state=1,
            #verbose=True
        )
        fitter = TSNEEmbedding(
            U[:,:1]*0.0001,
            affinities,
            n_jobs=16,
            random_state=1,
            #verbose=True
        )
        embeddingOPENTSNE = fitter.optimize(n_iter=250)
    else:
        tsne = TSNE(
            perplexity=perplexities[0],
            metric="cosine",
            n_jobs=8,
            random_state=42,
            verbose = True,
            n_components = 1,
            initialization = .0001 * U[:,:1],
        )
        embeddingOPENTSNE = tsne.fit(U)
        
    return embeddingOPENTSNE

def run_UMAP(U, n_neighbors=15):
    embeddingUMAP = UMAP(n_components=1, n_neighbors=n_neighbors).fit_transform(U)
    return embeddingUMAP

def benchmark_embeddings(S, bin_size=0):
    """ S is n_samples by n_features, optional bin_size to bin over features """
    
    if bin_size > 0:
        Sb = bin1d(S.T, bin_size).T
    else:
        Sb = S

    itest, itrain = split_testtrain(Sb.shape[1])
    
    model = Rastermap(smoothness=1, 
                        n_clusters=200, 
                        n_PCs=200, 
                        grid_upsample=10, 
                        alpha=1.).fit(Sb, itrain=itrain)

    timings, embeddings = [], []
    embeddings.append(model.embedding)
    timings.append(model.map_time)

    tic=time.time()
    embeddings.append(run_TSNE(model.U))
    timings.append(time.time()-tic)

    tic=time.time()
    embeddings.append(run_UMAP(model.U))
    timings.append(time.time()-tic)

    neighbor_scores = np.zeros((len(embeddings), 3))
    global_scores = np.zeros(len(embeddings))
    for i, emb in enumerate(embeddings):
        mnn, rho = embedding_score(model.X_test, emb)
        neighbor_scores[i] = mnn
        global_scores[i] = rho

    return neighbor_scores, global_scores, timings