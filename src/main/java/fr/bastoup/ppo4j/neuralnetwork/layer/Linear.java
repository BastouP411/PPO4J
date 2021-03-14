/*==============================================================================
 Copyright 2021 BastouP

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 =============================================================================*/

package fr.bastoup.ppo4j.neuralnetwork.layer;

import fr.bastoup.ppo4j.neuralnetwork.activation.Activation;
import fr.bastoup.ppo4j.neuralnetwork.activation.Identity;
import fr.bastoup.ppo4j.neuralnetwork.layer.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Linear implements Layer {

    private final int[] shape;
    private final Activation activation;
    private final INDArray weights;
    private final boolean bias;

    // Constructors

    public Linear(int n, int m, Activation activation, boolean bias) {
        this.bias = bias;
        this.shape = new int[] {n,m};

        if(bias)
            this.weights = Nd4j.rand(n, m);
        else
            this.weights = Nd4j.rand(n + 1, m);

        this.activation = activation;
    }

    public Linear(int n, int m, Activation activation) {
        this(n, m, activation, true);
    }

    public Linear(int n, int m) {
        this(n, m, new Identity(), true);
    }

    // Getters

    public int[] getShape() {
        return shape;
    }

    public Activation getActivation() {
        return activation;
    }

    public INDArray getWeights() {
        return weights;
    }

    public boolean isBias() {
        return bias;
    }

    // Methods

    @Override
    public INDArray activate(INDArray array) {
        INDArray col = weights.mmul(array);
        return activation.apply(col);
    }
}

