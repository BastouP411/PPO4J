package fr.bastoup.ppo4j.neuralnetwork.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class RELU implements Activation {

    @Override
    public INDArray apply(INDArray array) {
        return Transforms.relu(array, true);
    }

    @Override
    public INDArray derivative(INDArray array) {
        return Transforms.sign(array, true);
    }
}