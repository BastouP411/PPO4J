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

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Main {
    public static void main(String[] args) {
        Logger logger = LoggerFactory.getLogger(Main.class);
        INDArray line = Nd4j.ones(3);
        INDArray mat = Nd4j.create(new float[] {1,1,1,0,1,1,0,0,1}, new int[] {3,3});
        System.out.println(mat);
        System.out.println(line.reshape(1,3).mmul(mat).reshape(3));
    }
}
