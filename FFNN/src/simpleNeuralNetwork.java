import Jama.Matrix;
//import com.sleepycat.bind.tuple.TupleInput;
import javafx.scene.paint.Material;

import java.io.*;
import java.lang.reflect.Array;
import java.util.*;

public class simpleNeuralNetwork {

    private int numLayers;
    private ArrayList<layer> layers = new ArrayList<>();
    private static Vector<Integer> size_layer;

    public simpleNeuralNetwork(Vector<Integer> layerSizes) {
        this.size_layer = layerSizes;
        numLayers = layerSizes.size();
        this.layers.add(new layer(layerSizes.get(0), layerSizes.get(1), "input"));

        for (int ii = 1; ii < numLayers - 1; ii++) {
            this.layers.add(new layer(layerSizes.get(ii), layerSizes.get(ii + 1), "hidden"));

        }
        System.out.println("Neural Network Instantiated");
    }


    public Matrix sigmoid(Matrix z) {
        for (int ii = 0; ii < z.getRowDimension(); ii++)
            z.set(ii, 0, Math.tanh(z.get(ii, 0)));
        return z;
    }

    public Matrix sigmoidPrime(Matrix z) {
        for (int ii = 0; ii < z.getRowDimension(); ii++) {
            double temp = z.get(ii, 0);
            z.set(ii, 0, 1 - Math.tanh(temp) * Math.tanh(temp));
        }
        return z;
    }


    public Matrix feedForward(Matrix a) {
        for (int ii = 0; ii < numLayers - 1; ii++) {
            a = sigmoid(this.layers.get(ii).getWieghts().times(a).plus(this.layers.get(ii).getBiases()));
        }
        a = softmax(a);

        //printMatrix(a);
        return a;
    }


    public Matrix costDeriv(Matrix act, Matrix y) {
        double result = 0.0;
        for (int ii = 0; ii < 40; ii++) {
            result -= y.get(ii, 0) * Math.log(act.get(ii, 0));
        }
        Matrix out = new Matrix(1, 1, result);
        return out;
    }


    public ArrayList<ArrayList<Matrix>> backProp(Matrix x, Matrix y) {
        ArrayList<Matrix> nablaB = new ArrayList<>();
        ArrayList<Matrix> nablaW = new ArrayList<>();
        ArrayList<Matrix> Acts = new ArrayList<>(); //list of activiations starting with x
        Acts.add(x);
        ArrayList<Matrix> zs = new ArrayList<>(); //store all the intermidaite Z's

        for (int ii = 0; ii < numLayers - 1; ii++) {
            Matrix z = this.layers.get(ii).getWieghts().times(Acts.get(ii)).plus(this.layers.get(ii).getBiases());
            //printMatrix(z);
            zs.add(z);
            Acts.add(sigmoid(z));
        }
        Matrix poss = softmax(Acts.get(numLayers - 1));
        Matrix delta = poss.minus(y);
        //printMatrix(delta);//delta of the last layer;
        delta = delta.arrayTimes(sigmoidPrime(zs.get(numLayers - 2)));
        nablaB.add(delta); //I AM ADDING IN REVERSE ORDER
        nablaW.add(delta.times(Acts.get(numLayers - 2).transpose())); //CHECK ABOUT THE TRANSPOSING HERE...

        for (int ii = numLayers - 2; ii > 0; ii--) //from second last layer backwards
        {
            Matrix z = zs.get(ii - 1);
            Matrix sp = sigmoidPrime(z);
            delta = layers.get(ii).getWieghts().transpose().times(delta).arrayTimes(sp);
            //times sp should be EbyE not dot product
            nablaB.add(0, delta);
            nablaW.add(0, delta.times(Acts.get(ii - 1).transpose())); //fixed order of dot product
        }
        ArrayList<ArrayList<Matrix>> result = new ArrayList<>();
        result.add(nablaB);
        result.add(nablaW);
        return result;
    }

    public void update(ArrayList<Matrix> batchX, ArrayList<Matrix> batchY, double eta) //each tuple in batch is ii and ii+1
    {

        ArrayList<ArrayList<Matrix>> nablaB = new ArrayList<>();
        ArrayList<ArrayList<Matrix>> nablaW = new ArrayList<>();
        int trainning_size = batchX.size();
        for (int jj = 0; jj < trainning_size; jj++) {
            ArrayList<ArrayList<Matrix>> backPropOUT = new ArrayList<>();
            backPropOUT.addAll(backProp(batchX.get(jj), batchY.get(jj)));
            nablaB.add(backPropOUT.get(0));
            nablaW.add(backPropOUT.get(1));
        }

        for (int jj = 0; jj < trainning_size; jj++) {
            for (int ii = 0; ii < numLayers - 1; ii++) {
                Matrix tempW = this.layers.get(ii).getWieghts();
                Matrix tempB = this.layers.get(ii).getBiases();
                Matrix tw = nablaW.get(jj).get(ii);
                this.layers.get(ii).setWieghts(tw.times(eta / trainning_size).times(-1.0).plus(tempW));
                this.layers.get(ii).setBiases(nablaB.get(jj).get(ii).times(eta / trainning_size).times(-1.0).plus(tempB));
            }
        }
    }


    public void SGD(ArrayList<Matrix> trainingDataX, ArrayList<Matrix> trainingDataY, int epochs, int miniBatchSize, double eta, String model_output) {
        int n = trainingDataX.size();
        //Random cool = new Random();
        //cool.setSeed(System.currentTimeMillis());
//        for (int ii = 0; ii < trainingDataY.size(); ii++){
//            printMatrix(trainingDataY.get(ii));
//        }
        for (int jj = 0; jj < epochs; jj++) {
            ArrayList<Matrix> miniBatchX = new ArrayList<>();
            ArrayList<Matrix> miniBatchY = new ArrayList<>();
            //System.out.println(trainingDataY.size());
            //printMatrix(trainingDataY.get(10));
            update(trainingDataX, trainingDataY, eta);
            System.out.println("EPOCH:" + jj + " COMPLETE");
            if (jj % 100 == 0) {
                printMatrix(feedForward(trainingDataX.get(1)));
                System.out.print("\n");
                printMatrix(trainingDataY.get(1));
                Matrix Sum = new Matrix(1, 1);
                Sum.set(0, 0, 0);
                for (int ii = 0; ii < trainingDataX.size(); ii++) {
                    Sum.plusEquals(costDeriv(feedForward(trainingDataX.get(ii)), trainingDataY.get(ii)));
                }
                double average = Sum.get(0, 0) / trainingDataX.size();
                if (average < 0.0001) return;
                System.out.println("The cost:" + average);
                try {
                    PrintWriter fr = new PrintWriter(model_output);
                    for (int ii = 0; ii < size_layer.size(); ii++) {
                        fr.print(size_layer.get(ii));
                        fr.print(" ");
                    }
                    fr.print("\n");
                    for (int ii = 0; ii < numLayers - 1; ii++) {
                        for (int j = 0; j < layers.get(ii).getWieghts().getRowDimension(); j++) {
                            for (int k = 0; k < layers.get(ii).getWieghts().getColumnDimension(); k++) {
                                fr.print(layers.get(ii).getWieghts().get(j, k) + " ");
                            }
                            fr.print(layers.get(ii).getBiases().get(j, 0));
                            fr.print("\n");
                        }
                        fr.print("\n");
                    }
                    fr.close();
                } catch (IOException ie) {
                }
            }
        }
    }

    public static void printMatrix(Matrix a) {
        for (int ii = 0; ii < a.getRowDimension(); ii++) {
            for (int jj = 0; jj < a.getColumnDimension(); jj++) {
                System.out.printf("|" + a.get(ii, jj));
            }
            System.out.println("|");
        }

    }

    public Matrix softmax(Matrix lastout) {
        int n = size_layer.get(numLayers-1);
        Matrix result = new Matrix(n, 1);
        double sum = 0.0;
        for (int ii = 0; ii < n; ii++) {
            sum = sum + Math.exp(lastout.get(ii, 0));
        }
        for (int ii = 0; ii < n; ii++) {
            result.set(ii, 0, Math.exp(lastout.get(ii, 0)) / sum);
        }
        return result;
    }

    public ArrayList<ArrayList<Matrix>> pre_processing(String File_counting, String input, Integer gram){
        int result_size = size_layer.get(numLayers-1);
        int inputsize = size_layer.get(0);
        ArrayList<Matrix> data_test = new ArrayList<>();
        ArrayList<Matrix> result = new ArrayList<>();
        ArrayList<ArrayList<Matrix>> output = new ArrayList<>();
        Matrix sum = new Matrix(1, 1);
        sum.set(0, 0, 0.0);
        HashMap<Integer, Matrix> map = new HashMap<>();
        try{
            File folder = new File(input);
            File[] files = folder.listFiles();
            for (File file:files) {
                FileReader fr = new FileReader(file);
                BufferedReader br = new BufferedReader(fr);
                String temp;
                Vector<Integer> ids = new Vector<>();
                while ((temp = br.readLine()) != null) {
                    ids.clear();
                    Matrix out = new Matrix(result_size, inputsize, 0.0);
                    String[] token = temp.split(" ");
                    for (int ii = 0; ii <= gram; ii++) {
                        ids.add(Integer.parseInt(token[ii]));
                    }
                    out.set(ids.get(1), ids.get(2), 1.0);
                    if (!map.containsKey(ids.get(0))) {
                        map.put(ids.get(0), out);
                    } else {
                        map.replace(ids.get(0), map.get(ids.get(0)).plus(out));
                    }
                }
                fr.close();
            }
        }catch (IOException ie){
            System.out.println("Cannot read the file");
        }
        try {
            PrintWriter fr_out = new PrintWriter(File_counting);
            int a = 0;
            Boolean flag = false;
            for (Integer p : map.keySet()) {
                Matrix vec = new Matrix(inputsize, 1, 0.0);
                Matrix out = new Matrix(result_size, 1, 0.0);
                for (int i = 0; i < 40; i++) {
                    flag = false;
                    for (int j = 0; j < 40; j++) {
                        out.set(j, 0, map.get(p).get(i, j));
                        if (map.get(p).get(i, j) > 0.0) flag = true;
                    }
                    vec.set(p, 0, 1);
                    vec.set(i + 40, 0, 1);
                    if (flag) {
                        double sum_all = 0;
                        for (int ii = 0; ii < out.getRowDimension(); ii++) {
                            sum_all = sum_all + out.get(ii, 0);
                        }
                        fr_out.print(p+" "+i+" "+sum_all+"\n");
                        sum_all = 1 / sum_all;
                        System.out.println(sum_all);
                        out.timesEquals(sum_all);
                        //simpleNeuralNetwork.printMatrix(out);
                        data_test.add(vec.copy());
                        result.add(out.copy());
                        //simpleNeuralNetwork.printMatrix(result.get(a));
                        //System.out.println("before" + a);
                        a = a + 1;
                    }
                }
            }
            fr_out.close();
        }catch(IOException ie){System.out.println("Cannot write into file.");}
        output.add(data_test);
        output.add(result);
        return output;
    }

    public void train(String File_model, String File_counting, String File_read, Integer gram,double eta, int epoch, int minisize) {
        ArrayList<ArrayList<Matrix>> train_data;
        ArrayList<Matrix> train_input = new ArrayList<>();
        ArrayList<Matrix> train_output = new ArrayList<>();
        train_data = this.pre_processing(File_counting,File_read, gram);
        train_input = train_data.get(0);
        train_output = train_data.get(1);
        this.SGD(train_input,train_output, epoch, minisize, eta, File_model);
    }
}