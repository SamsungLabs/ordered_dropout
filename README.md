# Ordered Dropout

Implementation of Ordered Dropout (OD) for different type of NN layers in PyTorch.

The repo contains:
 
- layers: Linear, CNN, LSTM,
- simple example to showcase OD: [example.ipynb](example.ipynb).

### How to run

```
pip install -r requirements
jupyter notebook
```

### Reference

If you find this repo useful, please cite the paper that introduced notion of 
Ordered Dropout:
```
@article{horvath2021fjord,
  title={Fjord: Fair and accurate federated learning under heterogeneous targets with ordered dropout},
  author={Horvath, Samuel and Laskaridis, Stefanos and Almeida, Mario and Leontiadis, Ilias and Venieris, Stylianos and Lane, Nicholas},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={12876--12889},
  year={2021}
}
```
