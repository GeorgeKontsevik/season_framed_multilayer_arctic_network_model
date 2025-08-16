

## Remove this file

Once you are done adapting this template to your use, remove this file.

##
This will install the latest versions of `pre-commit` and `pytest`. If you need a specific version, modify the files.

You then install the [pre-commit](https://pre-commit.com/) hooks that are specified in the `.pre-commit-config.yaml`. You can remove or add the ones that you like. Notable hooks are:

- [Ruff](https://github.com/astral-sh/ruff-pre-commit) linting and formatting as a pre-commit hook.
- [Prettier](https://github.com/pre-commit/mirrors-prettier) for formatting multiple formats of files such as YML or JSON.
- [Out-of-the-box basic hooks](https://github.com/pre-commit/pre-commit-hooks) for a list of basic hooks.
You install the ones specified in `.pre-commit-config.yaml` by running:

```
pre-commit install
```


## After publication
When you make the code public or publish a paper, [update the citation file template](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files) in your repo: [`CITATION.cff`](CITATION.cff)

As best practice, place your code/data on [zenodo](https://zenodo.org). Follow the [tutorial to connect your github repo with zenodo](https://github.com/OpenScienceMOOC/Module-5-Open-Research-Software-and-Open-Source/blob/master/content_development/Task_2.md).


## Troubleshooting
If you have problems committing due to the linter complaining, try removing the following lines from `.pre-commit-config.yaml`:

```
      # Run the linter.
      - id: ruff
        types_or: [python, pyi, jupyter]
        args: [--fix]
```
