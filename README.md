# Iterative Archetypal Analysis (IterAA)

## Description

`IterAA` is a package that provides functionalities to conduct accelerated archetypal analysis via an iterative approach.

## Background
* Archetypal analysis is an unsupervised learning technique that uses a convex polytope to summarise multivariate data.
* The classical algorithm involves an alternating minimisation algorithm, which grows quadratically in complexity.
* An iterative approach could be implemented to accelerate the execution of the archetypal analysis algorithm.
* The acceleration achieved by the iterative approach is in addition to the acceleration as a result of the optimisation of other portions of the algorithm execution, as was typically done in the past.

## Features
* Implementation of an iterative approach to conduct archetypal analysis.
* Implementation of a parallelised iterative approach to conduct archetypal analysis.
* Utilisation of high-performance-computing cluster for parallelisation of individual archetypal analysis execution on data subsets.

## Installation

Use `pip` to install `IterAA`:

```bash
$ pip install iteraa
```

## Usage

```python
from iteraa import getExampleBlobData, ArchetypalAnalysis

X = getExampleBlobData()  # Replace with your data
numArch = 3
aa = ArchetypalAnalysis(nArchetypes=numArch)
aa.fit(X)
aa.archetypes  # Archetypes
aa.explainedVariance_  # Explained variance

# Simplex plot
archIDs=range(numArch)
aa.plotSimplex(aa.alfa, archIDs)

# Profile plots
featNames, featIDs = X.columns(), range(len(X.columns))
aa.plotProfile(featNames, featIDs, archIDs)
aa.plotRadarProfile(featNames, featIDs, archIDs)

# Most archetypal samples
aa._extractCloseMatch()
aa.plotCloseMatch(archIDs)
```

Check out the notebooks for demonstrations of the [iterative](https://github.com/Jon-Ting/iaa/blob/main/docs/iaaDemo.ipynb) and [parallel iterative](https://github.com/Jon-Ting/iaa/blob/main/docs/piaaDemo.ipynb) approaches.

## Documentation

Detailed [documentations](https://iaa.readthedocs.io/en/latest/) are hosted by `Read the Docs`.

## Contributing

`IterAA` appreciates your enthusiasm and welcomes your expertise!

Please check out the [contributing guidelines](https://github.com/Jon-Ting/iaa/blob/main/CONTRIBUTING.md) and [code of conduct](https://github.com/Jon-Ting/iaa/blob/main/CONDUCT.md). 
By contributing to this project, you agree to abide by its terms.

## License

The project was created by Jonathan Yik Chang Ting. It is licensed under the terms of the [MIT license](https://github.com/Jon-Ting/iaa/blob/main/LICENSE).

## Credits

The package was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
The code is developed based on the [code structure and functionalities for visualisation of the *archetypes.py* written by Benyamin Motevalli](https://researchdata.edu.au/archetypal-analysis-package/1424520), who in turn developed his code based on ["Archetypal Analysis" by Adele Cutler and Leo Breiman, Technometrics, November 1994, Vol.36, No.4, pp. 338-347](https://www.jstor.org/stable/1269949).

## Contact

Email: `Jonathan.Ting@anu.edu.au`/`jonting97@gmail.com`

Feel free to reach out if you have any questions, suggestions, or feedback.
