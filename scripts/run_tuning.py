from scape.scape_t.tuning import evolutionary_search
from scape import datasets

dev_corpus = {**{s:0 for s in datasets.Dev_simple}, **{s:1 for s in datasets.Dev_complex}}
best, history = evolutionary_search(dev_corpus, datasets.Dev_simple, datasets.Dev_complex, n_iter=100, n_offspring=10, scale=0.3)
print(f"Best ACC: {best['acc']:.4f}\nBest params: {best['params']}")
