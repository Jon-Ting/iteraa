# Iterative Archetypal Analysis (IAA)

## Description

`IAA` is a package that provides functionalities to conduct accelerated archetypal analysis via an iterative approach.

## Background
* Archetypal analysis ... TODO

## Features
* Utilisation of high-performance-computing cluster (under development) ... TODO

## Installation

Use `pip` or `conda` to install `IAA`:

```bash
$ pip install iaa
```

```bash
$ conda install -c conda-forge iaa
```

## Usage

```python
from iaa import ArchetypalAnalysis

X = getExampleData()  # Replace with your data
aa = ArchetypalAnalysis()
aa.fit(X)
```

Check out the notebooks for demonstrations of the [iterative]() and [parallel iterative]() approaches.

## Documentation

Detailed [documentations]() are hosted by `Read the Docs`.

## Contributing

`IAA` appreciates your enthusiasm and welcomes your expertise!

Please check out the [contributing guidelines]() and [code of conduct](). 
By contributing to this project, you agree to abide by its terms.

## License

The project was created by Jonathan Yik Chang Ting. It is licensed under the terms of the [MIT license]().

## Credits

The package was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
The code is developed based on the [code structure and functionalities for visualisation of the *archetypes.py* written by Benyamin Motevalli](https://researchdata.edu.au/archetypal-analysis-package/1424520), who in turn developed his code based on ["Archetypal Analysis" by Adele Cutler and Leo Breiman, Technometrics, November 1994, Vol.36, No.4, pp. 338-347](https://www.jstor.org/stable/1269949).

## Contact

Email: `Jonathan.Ting@anu.edu.au`/`jonting97@gmail.com`

Feel free to reach out if you have any questions, suggestions, or feedback.
