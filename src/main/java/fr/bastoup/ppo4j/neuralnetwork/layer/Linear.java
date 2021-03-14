package fr.bastoup.ppo4j.neuralnetwork.layer;

import fr.bastoup.ppo4j.neuralnetwork.activation.Activation;
import fr.bastoup.ppo4j.neuralnetwork.layer.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Linear implements Layer {

    private final int[] shape;
    private final INDArray weights;

    public Linear(int n, int m, Activation activation) {
        this.shape = new int[] {n,m};
        this.weights = Nd4j.rand(this.shape);
    }

    public int[] getShape() {
        return shape;
    }

    @Override
    public INDArray activate(INDArray array) {
        return null;
    }
}

