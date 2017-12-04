import Jama.Matrix;

import javax.lang.model.util.SimpleElementVisitor6;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;
import java.util.Vector;

public class nnMain {

    public static void main(String[] args) {
        Vector<Integer> testVec = new Vector<>();
        //the data shown below cannot be changed for the train based on our data structrue.
        int vocab_size = 40;
        int gram = 2;
        String File_model = "model.txt";
        String File_count = "counting.txt";
        String test_folder = "trainning_data";
        int minisize = 1000;
        //You can play with the following coefficients
        //eta : the trainning rate.
        //epoch : how long do you want your trainning to run.
        double eta = 0.13;
        int epoch = 10000;
        // You can change or add layers.
        testVec.add(vocab_size*gram); // Input layer
        testVec.add(120);             // Hidden layer (You can add layers with different sizes)
        testVec.add(vocab_size);      // Output Layer (Must be added at last

        simpleNeuralNetwork test = new simpleNeuralNetwork(testVec);
        //test.train(File_model,File_count, test_folder, gram, eta, epoch, minisize);
        test.pre_processing(File_count, test_folder,gram);
    }
}

