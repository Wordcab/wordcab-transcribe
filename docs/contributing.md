## Getting started

1. Ensure you have the `Hatch` installed (with pipx for example):

- [hatch](https://hatch.pypa.io/latest/install/)

2. Clone the repo

```bash
git clone
cd wordcab-transcribe
```

3. Install dependencies and start coding

```bash
hatch env create
```

4. Run tests

```bash
# Quality checks without modifying the code
hatch run quality:check

# Quality checks and auto-formatting
hatch run quality:format

# Run tests with coverage
hatch run tests:run
```

## Working workflow

1. Create an issue for the feature or bug you want to work on.
2. Create a branch using the left panel on GitHub.
3. `git fetch`and `git checkout` the branch.
4. Make changes and commit.
5. Push the branch to GitHub.
6. Create a pull request and ask for review.
7. Merge the pull request when it's approved and CI passes.
8. Delete the branch.
9. Update your local repo with `git fetch` and `git pull`.
