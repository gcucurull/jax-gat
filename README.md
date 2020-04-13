# Graph Attention Networks in JAX

This repository implements Graph Attention Networks (GATs) in JAX. The code contains the model definition of a main GAT model with two graph attention layers, following the model used in the paper [Graph Attention Networks](https://arxiv.org/pdf/1710.10903.pdf).

## Usage
Run 

```python train.py```

to train a model on the Cora dataset.

## Good to know
This repository implementents Graph Attention Networks, but it doesn't fully replicate the paper. For example, it doesn't include early stopping or model saving in the training loop.

## Cite
If you use this code in your research, please cite the paper:
```
@article{
  velickovic2018graph,
  title="{Graph Attention Networks}",
  author={Veli{\v{c}}kovi{\'{c}}, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Li{\`{o}}, Pietro and Bengio, Yoshua},
  journal={International Conference on Learning Representations},
  year={2018},
  url={https://openreview.net/forum?id=rJXMpikCZ},
  note={accepted as poster},
}
``` 