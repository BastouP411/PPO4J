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

package fr.bastoup.ppo4j.neuralnetwork;

import fr.bastoup.ppo4j.neuralnetwork.layer.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;

public class NeuralNetwork {
    private final Layer[] layers;

    public NeuralNetwork(Layer... layers) {
        this.layers = layers;
    }

    public INDArray forward(INDArray obs) throws WrongArraySizeException {
        long[] obsShape = obs.shape();
        if(obsShape.length != 1)
            throw new WrongArraySizeException("Provided array should be 1D but is " + obsShape[0] + "D.");

        INDArray current = obs.reshape(1, obsShape[0]);
        for(int i = 0; i < layers.length; i++) {
            current = layers[i].activate(current);
        }

        long[] outputShape = current.shape();
        return current.reshape(outputShape[1]);
    }
}
