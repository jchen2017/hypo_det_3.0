# coding=utf-8
#
# Released under MIT License
#
# Copyright (c) 2019, Jinying Chen
#
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

import numpy as np

class HypoglycemiaClassifier():

  def __init__(self, model, rules=True):
    self.model = model
    self.rules = rules


  def train(self, X, Y, exclusions=None):
    if exclusions is not None and self.rules:
      X_model = X[exclusions != 0]
      Y_model = Y[exclusions != 0]
    else:
      X_model = X
      Y_model = Y
    self.model.fit(X_model, Y_model)


  def test(self, X, exclusions=None):
    y_hat = []
    for i in range(X.shape[0]):
      if exclusions[i] == 0 and self.rules:
        y_hat.append(0)
      else:
        pred_y=0
        try:
          pred_y=self.model.predict(X[i].reshape(1, -1))
        except:
          pred_y=self.model.predict(X[i])
        print (pred_y)
        y_hat.append(pred_y[0])
          
    return np.array(y_hat)

