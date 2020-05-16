# Label Consistent KSVD algorithm (LC-KSVD)

## Description
Implementation of the Label consistent KSVD algorithm proposed by Zhuolin Jiang, Zhe Lin and Larry S. Davis.

This implementation is a translation of the matlab code released by the authors on /projectlcksvd.html.

The code has been extended in order to use the related method called Discriminative KSVD proposed by Zhang, Qiang and Li, Baoxin.


Forked from [https://github.com/Laadr/LC-KSVD](https://github.com/Laadr/LC-KSVD)

## Usage
### LC-KSVD1
```python
from scipy.io import loadmat
from sklearn.metrics import accuracy_score

from <some folder>.lc_ksvd.dksvd import DKSVD

file_path = 'path_to_data/featurevectors.mat'
data = loadmat(file_path)
lcksvd = DKSVD(timeit=True)
Dinit, Tinit_T, Winit_T, Q = lcksvd.initialization4LCKSVD(data['training_feats'], data['H_train'])
D, X, T, W = lcksvd.labelconsistentksvd1(data['training_feats'], Dinit, data['H_train'], Q, Tinit_T)
predictions, gamma = lcksvd.classification(D, W, data['testing_feats'])
print('\nFinal recognition rate for LC-KSVD1 is : {0:.4f}'.format(
    accuracy_score(np.argmax(data['H_test'], axis=0), predictions)))
```

### LC-KSVD2
```python
from scipy.io import loadmat
from sklearn.metrics import accuracy_score

from <some folder>.lc_ksvd.dksvd import DKSVD

file_path = 'path_to_data/featurevectors.mat'
data = loadmat(file_path)
lcksvd = DKSVD(timeit=True)
Dinit, Tinit_T, Winit_T, Q = lcksvd.initialization4LCKSVD(data['training_feats'], data['H_train'])

D, X, T, W = lcksvd.labelconsistentksvd2(data['training_feats'], Dinit, data['H_train'], Q, Tinit_T, Winit_T)
predictions, gamma = lcksvd.classification(D, W, data['testing_feats'])
print('\nFinal recognition rate for LC-KSVD2 is : {0:.4f}'.format(
    accuracy_score(np.argmax(data['H_test'], axis=0), predictions)))
```

### D-KSVD
```python
from scipy.io import loadmat
from sklearn.metrics import accuracy_score

from <some folder>.lc_ksvd.dksvd import DKSVD

file_path = 'path_to_data/featurevectors.mat'
data = loadmat(file_path)
lcksvd = DKSVD(timeit=True)
Dinit, Winit = lcksvd.initialization4DKSVD(data['training_feats'], data['H_train'])
predictions, gamma = lcksvd.classification(Dinit, Winit, data['testing_feats'])
print('\nFinal recognition rate for D-KSVD is : {0:.4f}'.format(
    accuracy_score(np.argmax(data['H_test'], axis=0), predictions)))
```

### Visualize learned representations
``` python
from scipy.io import loadmat
from sklearn.metrics import accuracy_score

from <some folder>.lc_ksvd.constants import PlotFilter
from <some folder>.lc_ksvd.dksvd import DKSVD
from <some folder>.lc_ksvd.utils.plot_tools import LearnedRepresentationPlotter


file_path = 'path_to_data/featurevectors.mat'
data = loadmat(file_path)
lcksvd = DKSVD(timeit=True)
Dinit, Tinit_T, Winit_T, Q = lcksvd.initialization4LCKSVD(data['training_feats'], data['H_train'])

D, X, T, W = lcksvd.labelconsistentksvd2(data['training_feats'], Dinit, data['H_train'], Q, Tinit_T, Winit_T)
predictions, gamma = lcksvd.classification(D, W, data['testing_feats'])
COLOURS = tuple(['r', 'g', 'b', 'orange'])
label_index = {0: 'Normal', 1: 'Benign', 2: 'In Situ', 3: 'Invasive'}

# plot_basic_figureb
LearnedRepresentationPlotter(predictions=predictions, gamma=gamma,label_index=label_index, custom_colours=COLOURS)(simple='')
# plot_colored_basic_figure
LearnedRepresentationPlotter(predictions=predictions, gamma=gamma,label_index=label_index, custom_colours=COLOURS)()
# plot_filtered_colored_image
LearnedRepresentationPlotter(predictions=predictions, gamma=gamma,label_index=label_index, custom_colours=COLOURS)(filter_by=PlotFilter.SHARED)
```


# Requirements
See requirements.txt

# Authors

The approximate KSVD algorithm included in the code has been written by nel215 (https://github.com/nel215/ksvd) (and very slighty modified).

Software translated and extended from matlab (http://users.umiacs.umd.edu/~zhuolin) to python by Adrien Lagrange (ad.lagrange@gmail.com), 2018.

Software fixed, refactored and modified to follow original matlab implementation by Giussepi Lopez (giussepexy at gmail.com), 2020.

# License

Distributed under the terms of the GNU General Public License 2.0.
