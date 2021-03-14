package fr.bastoup.ppo4j.neuralnetwork.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Identity implements Activation {

    @Override
    public INDArray apply(INDArray array) {
        return array.dup();
    }

    @Override
    public INDArray derivative(INDArray array) {
        return Nd4j.onesLike(array);
    }
}
