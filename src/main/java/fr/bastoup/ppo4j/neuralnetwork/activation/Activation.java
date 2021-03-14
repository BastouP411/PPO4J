package fr.bastoup.ppo4j.neuralnetwork.activation;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface Activation {
    INDArray apply(INDArray array);
    INDArray derivative(INDArray array);
}
