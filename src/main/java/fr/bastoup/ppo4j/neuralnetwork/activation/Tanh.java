package fr.bastoup.ppo4j.neuralnetwork.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class Tanh  implements Activation {

    @Override
    public INDArray apply(INDArray array) {
        return Transforms.tanh(array, true);
    }

    @Override
    public INDArray derivative(INDArray array) {
        return Nd4j.onesLike(array).sub(array.mul(array));
    }
}