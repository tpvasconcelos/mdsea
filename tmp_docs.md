## Add a new dataset

To add a new default 1D dataset to mdsea you first need to ensure
that the values you want to store have the expected requisits. ...

1. Go to the bottom of [fileman.py](mdsea/constants/fileman.py) and add
    a new variable declaring the name of your dataset in the format:
    ```
    <DATASETNAME>_DSNAME = '<dataset_name>'
    ```
    This means that if your dataset name has multiple words, the
    variable name should be the concatenation of these words without
    any spaces. On the other hand, the variable value should be the
    concatenation of these words separated by a hiphen (`_`). e.g.: If
    you were to add a dataset for the total energy of the system, it
    would look something like:
    ```python3
    TOTALENERGY_DSNAME = 'total_energy'
```

2. Go to [core.py](mdsea/core.py): `SysManager > __init__()` and look for
    ```python3

    # Datasets with shape=(STEPS,)
    self.potenergy_dsname = cfm.POTENERGY_DSNAME
    self.kinenergy_dsname = cfm.KINENERGY_DSNAME
    self.dists_dsname = cfm.DISTS_DSNAME
    self.temp_dsname = cfm.TEMP_DSNAME
    ...

    ```
    Here, you can add a class attribute with the name of the dataset.
    The attribute name should be the same as the one inheriting from
    [fileman.py](mdsea/constants/fileman.py) but lower case.

3. Finally, append your dataset name to the `self.oned_dsnames` list in
    [core.py](mdsea/core.py): `SysManager > __init__()`

4. Go to [simulator.py](mdsea/simulator.py): `_BaseSimulator > update_files()`
    and add the 1D array of values to the `values` list. Make sure it
    is in the same order as the `oned_dsnames` list (see step 3.)!
