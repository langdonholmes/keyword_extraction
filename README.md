# Keyword Extraction

This is a project for training a keyword extraction model. The model labels keyterms in OpenStax textbooks.

## 🚀 Quickstart

This project is built on spaCy projects. It can be used via the
[`spacy project`](https://spacy.io/api/cli#project) CLI. To find out
more about a command, add `--help`. For detailed instructions, see the
[usage guide](https://spacy.io/usage/projects).

1. **Clone** the project.
   ```bash
   python -m spacy project clone --repo https://github.com/aloe/deidentification-pipeline
   ```
2. **Run a command** defined in the `project.yml`.
   ```bash
   python -m spacy project run init
   ```
3. **Run a workflow** of multiple steps in order.
   ```bash
   python -m spacy project run all
   ```
