import cPickle
from sklearn.manifold import TSNE

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

file_dir = '/cvgl2/u/rvolpi/experiments/SYNTHIA-SEQS-01-DAWN/256/features_pretrain.pkl'

with open(file_dir,'r') as f:
	source_features, target_features, generated_features = cPickle.load(f)

generated_features = generated_features[0]

source_features = source_features[:300]
target_features = target_features['SYNTHIA-SEQS-01-NIGHT'][:300]
generated_features = generated_features[:300]

model = TSNE(n_components=2, random_state=0)

f, (ax1,ax2) = plt.subplots(1,2,sharey=True)

ax1.set_facecolor('white')
ax2.set_facecolor('white')
#~ ax3.set_facecolor('white')
	    
print 'Compute t-SNE 1.'
TSNE_hA = model.fit_transform(np.vstack((source_features, generated_features)))
ax1.scatter(TSNE_hA[:,0], TSNE_hA[:,1], c = np.hstack((np.ones((len(generated_features),)), 2 * np.ones((len(source_features),)))), s=3, cmap = mpl.cm.jet, alpha=0.5)

print 'Compute t-SNE 2.'
TSNE_hA = model.fit_transform(np.vstack((source_features, target_features)))
ax2.scatter(TSNE_hA[:,0], TSNE_hA[:,1], c = np.hstack((np.ones((len(generated_features),)), 2 * np.ones((len(source_features),)))), s=3, cmap = mpl.cm.jet, alpha=0.5)


#~ print 'Compute t-SNE 3.'
#~ TSNE_hA = model.fit_transform(np.vstack((source_features, target_features['SYNTHIA-SEQS-01-DAWN'])))
#~ ax3.scatter(TSNE_hA[:,0], TSNE_hA[:,1], c = np.hstack((np.ones((len(generated_features),)), 2 * np.ones((len(source_features),)))), s=3, cmap = mpl.cm.jet, alpha=0.5)

plt.title('DAWN vs GENERATED - DAWN vs NIGHT')
plt.show()
