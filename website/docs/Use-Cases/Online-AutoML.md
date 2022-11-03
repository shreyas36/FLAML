# Online AutoML

## Overview

[`flaml.AutoVW`](../reference/onlineml/autovw) is a class for doing online AutoML with [Vowpal Wabbit learners](https://github.com/VowpalWabbit/vowpal_wabbit/tree/master/python) in FLAML. It can be used to tune both conventional numerical and categorical hyperparameters, such as learning rate, and hyperparameters for featurization choices, such as the namespace (a namespace is a group of features) interactions in Vowpal Wabbit.


An example of online namespace interactions tuning with `flaml.AutoVW`:

```python
# require: pip install flaml[vw]
from flaml import AutoVW
'''create an AutoVW instance for tuning namespace interactions'''
autovw = AutoVW(max_live_model_num=5, search_space={'interactions': AutoVW.AUTOMATIC})
```

An example of online tuning of both namespace interactions and learning rate in VW:

```python
# require: pip install flaml[vw]
from flaml import AutoVW
from flaml.tune import loguniform
''' create an AutoVW instance for tuning namespace interactions and learning rate'''
# set up the search space and init config
search_space_nilr = {'interactions': AutoVW.AUTOMATIC, 'learning_rate': loguniform(lower=2e-10, upper=1.0)}
init_config_nilr = {'interactions': set(), 'learning_rate': 0.5}
# create an AutoVW instance
autovw = AutoVW(max_live_model_num=5, search_space=search_space_nilr, init_config=init_config_nilr)
```

A user can use the resulting AutoVW instances `autovw` in a similar way to a vanilla Vowpal Wabbit instance, i.e., `pyvw.vw`, to perform online learning by iteratively calling its `predict(data_example)` and `learn(data_example)` functions at each data example.

For more examples, please check out
[AutoVW notebook](https://github.com/microsoft/FLAML/blob/main/notebook/autovw.ipynb).