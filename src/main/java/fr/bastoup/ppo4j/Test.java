package fr.bastoup.ppo4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Test {
    public static void main(String[] args) {
        INDArray arr = Nd4j.ones(3);
        arr.put(0,0, -1);
        System.out.println(arr);
    }
}
