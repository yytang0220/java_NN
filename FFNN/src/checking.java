import Jama.Matrix;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Vector;

public class checking {

    private ArrayList<layer> layers = new ArrayList<>();
    private Matrix count;
    Vector<Integer> model_structre = new Vector<>();
    int numLayers = 0;
    int gram = 2;
    //During the initialization, set the FFNN based on pre_trained model, and import the counting file.
    public checking(int n){
        gram = n;
        try{
            System.out.println("Loading File");
            FileReader fr_model = new FileReader("model.txt");
            FileReader fr_count = new FileReader("counting.txt");
            BufferedReader br_model = new BufferedReader(fr_model);
            BufferedReader br_counting = new BufferedReader(fr_count);
            String temp;
            String[] tokens;
            //Read the model structure from the pre-trained file.
            temp = br_model.readLine();
            tokens = temp.split(" ");
            for (String token : tokens) {
                model_structre.add(Integer.parseInt(token));
            }
            numLayers = model_structre.size();
            for (int ii = 0; ii < numLayers - 1; ii++) {
                this.layers.add(new layer(model_structre.get(ii), model_structre.get(ii + 1), "hidden"));
            }

            int current_layer = 0;
            Matrix weight = new Matrix(model_structre.get(current_layer+1),model_structre.get(current_layer));
            Matrix bias = new Matrix(model_structre.get(current_layer+1),1);
            int line = 0;
            while ((temp = br_model.readLine())!= null){
                tokens = temp.split(" ");
                if (tokens.length <= 1){
                    layers.get(current_layer).setWieghts(weight.copy());
                    layers.get(current_layer).setBiases(bias.copy());
                    if (current_layer < numLayers - 2) {
                        current_layer += 1;
                        weight = new Matrix(model_structre.get(current_layer + 1), model_structre.get(current_layer));
                        bias = new Matrix(model_structre.get(current_layer + 1), 1);
                    }
                    line = 0;
                    continue;
                }
                for (int ii = 0; ii < tokens.length; ii++){
                    if (ii == model_structre.get(current_layer)) bias.set(line,0,Double.parseDouble(tokens[ii]));
                    else weight.set(line, ii, Double.parseDouble(tokens[ii]));
                }
                line = line + 1;
            }
            layers.get(current_layer).setWieghts(weight.copy());
            layers.get(current_layer).setBiases(bias.copy());

            count = new Matrix(model_structre.get(numLayers-1),model_structre.get(numLayers-1));
            while ((temp = br_counting.readLine()) != null){
                tokens = temp.split(" ");
                Vector<Integer> IDS = new Vector<>();
                for (int ii = 0; ii <= gram; ii++){
                    IDS.add((int)Double.parseDouble(tokens[ii]));
                }
                count.set(IDS.get(0),IDS.get(1),IDS.get(2));
            }
            System.out.println("Loading finished");
        }catch(IOException ie){
            System.out.println("Loading model error!");
        }
    }

    //Do the checking here. If the average possibility is low then return false. If

     public Boolean check_sentence(ArrayList<Integer[]> IDS){
        ArrayList<Boolean> out = new ArrayList<>();
        Matrix input = new Matrix(gram*model_structre.get(numLayers-1),1);
        double sum = 0.0;
        System.out.println(IDS.size());
        for (int jj = 0; jj < IDS.size(); jj++) {
            Integer[] V = IDS.get(jj);
            Matrix POS;
            for (int ii = 0; ii < gram; ii++) {
                input.set(V[ii] + ii * model_structre.get(numLayers - 1), 0, 1);
            }
            System.out.println((count.get(V[0],V[1])));
            if (count.get(V[0],V[1]) < 20) {return false;}
            POS = feedForward(input);
            simpleNeuralNetwork.printMatrix(POS);
            System.out.println("\n"+POS.get(V[gram],0));
            if (POS.get(V[gram],0) < 0.001){return false;}
            sum += POS.get(V[gram],0);
        }
        sum = sum/IDS.size();
        return true;
     }

     //The methods below copys from simpleNeuralNetwork, in order to run the FF process.

    public Matrix feedForward(Matrix a) {
        for (int ii = 0; ii < numLayers - 1; ii++) {
            a = sigmoid(this.layers.get(ii).getWieghts().times(a).plus(this.layers.get(ii).getBiases()));
        }
        a = softmax(a);
        return a;
    }

    public Matrix sigmoid(Matrix z) {
        for (int ii = 0; ii < z.getRowDimension(); ii++)
            z.set(ii, 0, Math.tanh(z.get(ii, 0)));
        return z;
    }

    public Matrix softmax(Matrix lastout) {
        int n = model_structre.get(numLayers-1);
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
}
