package fr.bastoup.ppo4j.neuralnetwork.layer;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface Layer {
    INDArray activate(INDArray array);
}
