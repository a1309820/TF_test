import pandas as pd
import numpy as np
import sklearn as sk
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.6)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn');


df = pd.DataFrame(np.concatenate((X,y)))
print(df)

training_set = tf.contrib.learn.extract_pandas_matrix(X)



print("Complete")