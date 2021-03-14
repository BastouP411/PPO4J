package fr.bastoup.ppo4j.neuralnetwork.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class Sigmoid implements Activation {

    @Override
    public INDArray apply(INDArray array) {
        return Transforms.sigmoid(array, true);
    }

    @Override
    public INDArray derivative(INDArray array) {
        return Transforms.sigmoidDerivative(array, true);
    }
}