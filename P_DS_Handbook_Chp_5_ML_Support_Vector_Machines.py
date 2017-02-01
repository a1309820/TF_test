

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns; sns.set()


# Baysian classification was a generative classification. Whereas Support Vector Machines are discriminative classifications; we simply find a line or curve (2-d) or manifold (n-dimensional) that divides that classes from each other.

# In[4]:

from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.6)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn');


# In[ ]:

# there is more than one line that can divide the classes


# In[9]:

xfit = np.linspace(-1,3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plt.plot([0.6], [2.1], 'x', color='red', markeredgewidth=2, markersize=10)

for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
    plt.plot(xfit, m* xfit +b, '-k')
plt.xlim(-1, 3.5);


# SVM: draw around each line a margin of some width, up to the nearest point.  The line with the maximum margin is called the maximum margin estimator.

# In[11]:

xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0],X[:, 1], c=y, s=50, cmap='autumn')

for m, b, d in [( 1.0, 0.65, 0.33),
                ( 0.5, 1.60, 0.55),
                (-0.2, 2.90, 0.20)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit, yfit-d, yfit+d, edgecolor='none', color='#AAAAAA', alpha=0.4)
    
plt.xlim(-1,3.5) 


# ## Fitting a SVM

# In[12]:

from sklearn.svm import SVC # "Support Vector Classifier"
model = SVC(kernel='linear', C=1E10)
model.fit(X,y)


# In[29]:

def plot_svc_decision_function(model, ax=None, plot_support=True):
    if ax is None:
        ax=plt.gca()
    xlim=ax.get_xlim()
    ylim=ax.get_ylim()
    
    # create grid to evaluate the model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y,x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot the decision boundary and margins
    ax.contour(X, Y, P, colors='k', levels=[-1,0,1], alpha=0.5,
              linestyles=['--','-','--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:,0],
                  model.support_vectors_[:,1],
                  s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


# In[30]:

plt.scatter(X[:,0], X[:,1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(model);


# In[31]:

model.support_vectors_


# In[38]:

def plot_svm(N=10, ax=None):
    X, y = make_blobs(n_samples=200, centers=2, random_state=0, cluster_std=0.60)
    X=X[:N]
    y=y[:N]
    model = SVC(kernel='linear', C=1E10)
    model.fit(X, y)
    
    ax = ax or plt.gca()
    ax.scatter(X[:,0], X[:,1], c=y, s=50, cmap='autumn')
    ax.set_xlim(-1,4)
    ax.set_ylim(-1,6)
    plot_svc_decision_function(model, ax)
    
fig, ax = plt.subplots(1,2, figsize=(16,6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
for axi, N in zip(ax, [60, 120]):
    plot_svm(N, axi)
    axi.set_title('N={0}'.format(N))


# In[40]:

from ipywidgets import interact, fixed
interact(plot_svm, N=[10,200], ax=fixed(None))


# ## Beyond linear boundaries
# Using kernels

# In[51]:

# non-linearly separable data
from sklearn.datasets.samples_generator import make_circles
X, y = make_circles(100, factor=0.1, noise=0.1)

clf = SVC(kernel='linear').fit(X,y)

plt.scatter(X[:,0], X[:,1], c=y, cmap='autumn')
plot_svc_decision_function(clf, plot_support=False);


# In[49]:

# try a radial basis function
r = np.exp(-(X**2).sum(1))


# In[55]:

from mpl_toolkits import mplot3d

def plot_3D(elev=30, azim=30, X=X, y=y):
    ax = plt.subplot(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=50, cmap='autumn') 
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('r')
    
interact(plot_3D, elev=[-90, 90], azip=(-180, 180),
         X=fixed(X), y=fixed(y));


# In[57]:

# use a rbf kernel
clf = SVC(kernel='rbf', C=1E6)
clf.fit(X,y)


# In[58]:

plt.scatter(X[:,0],X[:,1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(clf)
plt.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1],
           s=300, lw=1, facecolors='none');


# ## Tuning the SVM: Softening the margins
# Not all data are perfectly seperatable; so soften the margins.  Softness is controlled by the tuning parameter C.  The larger the C, the harder the margin.

# In[60]:

X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=1.2)
plt.scatter(X[:,0],X[:,1], c=y, s=50, cmap='autumn');


# In[61]:

X,y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=0.8)
fig, ax = plt.subplots(1, 2, figsize=(16,6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

for axi, C in zip(ax, [10.0, 0.1]):
    model = SVC(kernel='linear', C=C).fit(X,y)
    axi.scatter(X[:,0],X[:,1], c=y, s=50, cmap='autumn')
    plot_svc_decision_function(model, axi)
    axi.scatter(model.support_vectors_[:,0],
               model.support_vectors_[:,1],
               s=300, lw=1, facecolors='none');
    axi.set_title('C={0:.1f}'.format(C), size=14)


# ## Example: Facial Recognition

# In[83]:

from sklearn.datasets import fetch_lfw_people 
faces = fetch_lfw_people(min_faces_per_person=60) 
print(faces.target_names) 
print(faces.images.shape)


# In[84]:

fig, ax = plt.subplots(5,5)
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[],
           xlabel=faces.target_names[faces.target[i]])


# In[85]:

from sklearn.svm import SVC
from sklearn.decomposition import RandomizedPCA
from sklearn.pipeline import make_pipeline

pca = RandomizedPCA(n_components=150, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)

#make training and validation
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target, random_state=42)


# In[86]:

# grid search to find the best model
from sklearn.model_selection import GridSearchCV
param_grid={'svc__C':[1,5,10,50],
           'svc__gamma':[0.0001, 0.0005, 0.001, 0.005]}
grid = GridSearchCV(model, param_grid)
get_ipython().magic(u'time grid.fit(Xtrain, ytrain)')
print(grid.best_params_)


# In[87]:

model = grid.best_estimator_
yfit = model.predict(Xtest)


# In[91]:

fig, ax = plt.subplots(7,7)
for i, axi in enumerate(ax.flat):
    axi.imshow(Xtest[i].reshape(62,47), cmap='bone')
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(faces.target_names[yfit[i]].split()[-1],
                  color='black' if yfit[i]==ytest[i] else 'red')
    fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14);


# In[89]:

from sklearn.metrics import classification_report
print(classification_report(ytest, yfit, target_names=faces.target_names))


# In[101]:

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, yfit)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=faces.target_names,
            yticklabels=faces.target_names)



# Look at http://opencv.org/ for more details and features on image processing.
