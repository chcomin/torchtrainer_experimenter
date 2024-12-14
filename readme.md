# Torchtrainer experimeter

Experiment template for training neural networks using the [torchtrainer package](https://github.com/chcomin/torchtrainer).

Use the following commands to set up the template:

```bash
# Clone the torchtrainer repository
git clone https://github.com/chcomin/torchtrainer.git
# Install torchtrainer as an editable package in a conda environment
pip install --no-build-isolation --no-deps -e torchtrainer
# Clone this repository
git clone https://github.com/chcomin/torchtrainer_experimenter.git
```

If you do not use conda, replace the pip install command by

```pip install -e torchtrainer```

By the way, if you use VSCode it might be necessary to add the torchtrainer directory to python.analysis.extraPaths for proper linter support.