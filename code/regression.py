# -*- coding: utf-8 -*-
# @Author: TomLotze
# @Date:   2020-09-18 11:18
# @Last Modified by:   TomLotze
# @Last Modified time: 2020-10-15 21:47

import torch
import torch.nn as nn


class Regression(nn.Module):
  """
  This class implements a Multi-layer Perceptron in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized a Regression object can perform forward.
  """

  def __init__(self, n_inputs, n_hidden, dropout_percentages, n_classes=1,
               neg_slope=0.02, batchnorm=True):
    """
    Initializes Regression object.

    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the Regression
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the Regression
      neg_slope: negative slope parameter for LeakyReLU
    """

    super(Regression, self).__init__()

    assert len(dropout_percentages) == len(n_hidden)

    layer_list = []
    if n_hidden:
      for nr_nodes, drop_perc in zip(n_hidden, dropout_percentages):
        layer_list.append(nn.Linear(n_inputs, nr_nodes))
        layer_list.append(nn.Dropout(p=drop_perc))
        layer_list.append(nn.BatchNorm1d(nr_nodes))
        layer_list.append(nn.LeakyReLU(neg_slope))


        n_inputs = nr_nodes
    layer_list += [nn.Linear(n_inputs, n_classes)]

    self.layers = nn.ModuleList(layer_list)

    print(self.layers)


  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed
    through several layer transformations.

    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    """

    for layer in self.layers:
      x = layer(x)

    return x



