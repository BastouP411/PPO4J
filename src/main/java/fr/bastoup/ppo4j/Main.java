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

package fr.bastoup.ppo4j;

import fr.bastoup.ppo4j.neuralnetwork.NeuralNetwork;
import fr.bastoup.ppo4j.neuralnetwork.WrongArraySizeException;
import fr.bastoup.ppo4j.neuralnetwork.activation.RELU;
import fr.bastoup.ppo4j.neuralnetwork.activation.Sigmoid;
import fr.bastoup.ppo4j.neuralnetwork.layer.Linear;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Main {
    public static void main(String[] args) throws WrongArraySizeException {
        NeuralNetwork nn = new NeuralNetwork(
                new Linear(10, 20, new Sigmoid()),
                new Linear(20, 30, new RELU())
        );
        INDArray inp = Nd4j.rand(10);
        System.out.println(inp);
        INDArray outp = nn.forward(inp);
        System.out.println(inp);
        System.out.println(outp);
    }
}
