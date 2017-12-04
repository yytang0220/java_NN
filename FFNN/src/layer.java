import java.util.ArrayList;
import java.util.Random;
import java.util.Vector;

import Jama.Matrix;

public class layer {

    private Matrix wieghts;
    private Matrix biases;
    private int neurons;
    private String type;


    public layer(){
        System.out.println("NO DEFUALT CONSTRUCTOR - USE OTHER ONE");
    }


    //initalizes layer with basic features
    public layer(int n, int nextSize, String t)
    {

        switch (t)
        {
            case "hidden":
            case "input":
            case "output":
                type = t;
                break;
            default:
                System.out.println("INVALID TYPE - PLEASE TRY AGAIN");
                type = "invalid";
                break;
        }

        this.neurons = n;
        Matrix Twieghts = new Matrix(nextSize,n);
        Matrix Tbiases = new Matrix(nextSize,1);

        Random temp = new Random();
        temp.setSeed(346572383); //setting a seed for consistant

        for(int ii = 0; ii < n; ii++)
        {
            for(int jj = 0; jj < nextSize; jj++)
            {
                Tbiases.set(jj,0, 0);
                Twieghts.set(jj, ii, temp.nextGaussian());
            }
        }

        wieghts = Twieghts;
        biases = Tbiases;

    }


    //Getters
    public int getSize(){return this.neurons;}
    public Matrix getBiases(){return this.biases;}
    public Matrix getWieghts(){return this.wieghts;}
    public String getType(){return type;}

    //Setters
    public void setBiases(Matrix b) {this.biases = b;}
    public void setWieghts(Matrix w){this.wieghts = w;}
    //No wieght or type setters becuase that could break the network


    public void setAsInput(layer input,Matrix x)
    {
        if(input.getType() == "input") input.setBiases(x);
        else System.out.println("NOT INPUT LAYER");
    }
}


