package datayield;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

import java.io.ByteArrayOutputStream;
import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;


public class DyNnArray {

    private double model[][][];
    private double[][][] weights;
    private double[][][] weightsNew;

    private String currentCategory = "";


    ArrayList modelConstructor = new ArrayList();


    private double errors[];
    private double initialError = 0;
    private double totalError = 0;

    private double target[];
    private double inputs[];
    private int epochs = 100;
    private int loopsPerImage = 1;
    private int samples = 1000;
    private double learningRate = 0.01;
    private double decay = 0.9999;

    /*

    y values of the model:

    0 = net
    1 = output
    2 = saved value for next backpropagation step
    3 = saved value for next backpropagation step

     */



    public void addInputLayer(int size){
        modelConstructor.add(size);
    }

    public void addHiddenLayer(int size){
        modelConstructor.add(size);
    }

    public void constructModel(){

        int s = modelConstructor.size();

        model = new double[s][][];
        weights = new double[s-1][][];
        weightsNew = new double[s-1][][];

        int xBefore = 0;

        for(int i = 0; i < s; i++){
            int x = (int) modelConstructor.get(i);
            double[][] table = new double[x][5];
            model[i] = table;

            if(i > 0){
                double[][] wT = new double[x][xBefore];
                weights[i-1] = wT;
                double[][] wTN = new double[x][xBefore];
                weightsNew[i-1] = wTN;
            }

            xBefore = x;

        }

        // set the error array equal to last layer length
        int t = weights[weights.length - 1].length;
        errors = new double[t];


        System.out.println("Model has number of layers: " + model.length);

        for(int i = 0; i < model.length; i++){
            System.out.println("Layer " + i + " has number of neurons: " +model[i].length);
        }


    }



    private String getRandomPath(String basePath, String f){

        try {

            basePath += f;

            currentCategory = f;
            //Log.debug("base path", basePath);

            File dir = new File(basePath);
            File[] files = dir.listFiles();
            Random rand = new Random();
            File file = files[rand.nextInt(files.length)];
            String r = file.getPath();
            return r;
        }catch(Exception x){
            System.out.println(x.toString());
            System.exit(1);
        }
        return null;
    }


    public double[] extractBytes (String ImageName) {

        try {

            BufferedImage bImage = ImageIO.read(new File(ImageName));

            //System.out.println(bImage.getHeight());
            int x = bImage.getHeight();
            //System.out.println(bImage.getWidth());
            int y = bImage.getWidth();
            double xy;
            double[] data = new double[y * x];
            int i = 0;

            for (int iy = 0; iy < y; iy++) {
                for (int ix = 0; ix < x; ix++) {
                    //System.out.println(bImage.getRGB(ix, iy));

                    xy = bImage.getRGB(ix, iy);
                    xy = xy * -1;
                    xy = xy / 16777216;

                    data[i] = xy;
                    i += 1;
                }
            }

            return data;

        }catch(Exception x){

            System.out.println("ERROR: " + x.toString());

        }

        return null;
    }

//    private double[] getInput(String path){
//
//        double retArr[];
//
//        try{
//            //byte[] fileContent = extractBytes (path);
//            //retArr = new double[fileContent.length];
//
//            retArr = extractBytes (path);
//
//            for(int i = 0; i < fileContent.length; i++){
//
//                //System.out.println(fileContent[i]);
//                Byte b = new Byte(fileContent[i]);
//                int x = b.intValue();
//                x = x + 128;
//
//                if (x  < 0){
//                    System.out.println(x);
//                }
//
//                if (x  > 256){
//                    System.out.println(x);
//                }
//
//                double d = (double) x;
//                d = d/ 256;
//                //System.out.println(d);
//                retArr[i] =  d;
//            }
//
//            return retArr;
//
//        }catch(Exception x){
//            Log.error("getInput()", x.toString());
//        }
//        return null;
//    }


    private double[] getTarget(String f){

        double[] ret = {0,0,0,0,0,0,0,0,0,0};
        int p = Integer.decode(f);
        ret[p] = 1;
        return ret;
    }


    public void testImage(){

        String basePath = "C:\\Users\\alex\\IdeaProjects\\DyNeuralNet\\Data\\training\\";

        String f = Long.toString(Math.round(Math.random() * 9));

        System.out.println(f);


        String randomPath = getRandomPath(basePath, f);

        //double[] inputArr = getInput(randomPath);

        try {
            double[] inputArr = extractBytes(randomPath);


            int i = 0;
            for (int y = 0; y < 28; y++) {

                String s = "";
                String sf = "";
                for (int x = 0; x < 28; x++) {

                    if (inputArr[i] > 0.6) {
                        s = "x";
                    } else {
                        s = " ";
                    }

                    //System.out.println(inputArr[i]);

                    i += 1;
                    sf += s;
                }
                System.out.println(sf + "|");

            }
        }catch(Exception x){
            //System.out.println("ERROR " + x.toString());
        }


    }
    public void train(){

        constructModel();
        initializeWeights();

        //target

        for(int e = 0; e < epochs; e++){
            for(int i = 0; i < samples; i++){

                String basePath = "C:\\Users\\alex\\IdeaProjects\\DyNeuralNet\\Data\\training\\";

                String f = Long.toString(Math.round(Math.random() * 9));

                String randomPath = getRandomPath(basePath, f);

                //double[] inputArr = getInput(randomPath);
                double[] inputArr = extractBytes(randomPath);
                target = getTarget(f);

                trainOneSample(inputArr);

                learningRate = learningRate * decay;

            }
            // make a few tests
            for(int z = 0; z < 10; z++){

                String basePathT = "C:\\Users\\alex\\IdeaProjects\\DyNeuralNet\\Data\\training\\";
                String f = Long.toString(Math.round(Math.random() * 9));
                String randomPath = getRandomPath(basePathT, f);
                double[] inputArr = extractBytes(randomPath);
                predictOneImage(inputArr);

            }

        }



    }



    private void initializeWeights(){

        // initialize weights
        for(int c = 0; c < weights.length; c++){
            for(int y = 0; y < weights[c].length; y++) {
                for(int z = 0; z < weights[c][y].length; z++) {
                    weights[c][y][z] = Math.random() * 0.1;
                    //weights[c][y][z] = 0.00001;
                }
            }
        }
    }

    private void setInputVariables(double inputArr[]){

        // set input values
        for(int k = 0; k < model[0].length; k++){
            model[0][k][1] = inputArr[k];
            //Log.debug("input", Double.toString( model[0][k][1]));
        }

    }



    private void predictOneImage(double[] arr){

        setInputVariables(arr);

        for(int layer = 0; layer < model.length; layer++){

            // ignore first layer as only input
            if(layer > 0){

                // for each neuron in the layer
                for(int x = 0; x <  model[layer].length; x++){

                    //for each incoming connected weight / incoming neuron
                    double net = 0;
                    for(int n = 0; n <  model[layer-1].length; n++){
                        // grab weight relative to previous layer
                        // grab the value from previous layer
                        double tst =  model[layer-1][n][1];
                        //Log.debug("input value", Double.toString(tst));
                        net += weights[layer - 1][x][n] * tst;
                    }

                    //System.exit(0);

                    model[layer][x][0] = net;
                    //Log.debug("net value", Double.toString(model[layer][x][0]));
                    model[layer][x][1] = logisticFunction(net);
                    //Log.debug("output value", Double.toString(model[layer][x][1]));
                }

            }
        }


        double one[] = new double[10];

        for(int t = 0; t < 10; t++){
            one[t] = model[model.length - 1][t][1];
        }

        String statement = "========== ";
        statement += " category: " + currentCategory;
        statement += " prediction: " + oneHot(one);
        System.out.println(statement);
    }


    private void trainOneSample(double inputArr[]){



        setInputVariables(inputArr);


        // do all training loopsPerImage
        for(int i = 0; i < loopsPerImage; i++){


            //Log.debug("-------------", "-----------------------------");

            // forward pass
            // go forward through all layers
            for(int layer = 0; layer < model.length; layer++){

                // ignore first layer as only input
                if(layer > 0){

                    // for each neuron in the layer
                    for(int x = 0; x <  model[layer].length; x++){

                        //for each incoming connected weight / incoming neuron
                        double net = 0;
                        for(int n = 0; n <  model[layer-1].length; n++){
                            // grab weight relative to previous layer
                            // grab the value from previous layer
                            double tst =  model[layer-1][n][1];
                            //Log.debug("input value", Double.toString(tst));
                            net += weights[layer - 1][x][n] * tst;
                        }

                        //System.exit(0);

                        model[layer][x][0] = net;
                        //Log.debug("net value", Double.toString(model[layer][x][0]));
                        model[layer][x][1] = logisticFunction(net);
                        //Log.debug("output value", Double.toString(model[layer][x][1]));
                    }

                }
            }



            // calculate error
            // for each neuron in final layer
            totalError = 0;
            for(int x = 0; x <  model[model.length-1].length; x++){
                errors[x] = target[x] - model[model.length-1][x][1];
                errors[x] = 0.5 * Math.pow(errors[x], 2);
                totalError += errors[x];
                //Log.important("Error now", Double.toString(errors[x]));
            }
            //Log.important("Total error now", Double.toString(totalError) );

            if(i == 0){
                initialError = totalError;
            }


            // backwards
            // loop through layer backwards
            for(int layer = model.length-1; layer > 0; layer--){

                // special case last layer
                if(layer == model.length-1){

                    // for each neuron in the layer
                    for(int x = 0; x <  model[layer].length; x++) {

                        double out = model[layer][x][1];
                        double in = model[layer][x][0];

                        //Log.debug("final layer in", Double.toString(in));
                        //Log.debug("final layer out", Double.toString(out));

                        double p1 = out - target[x];
                        double p2 = this.logisticDerivative(out);

                        //Log.debug("final layer p1", Double.toString(p1));



                        // now loop through all neuron of previous layer
                        for(int pL = 0; pL <  model[layer - 1].length; pL++) {

                            //Log.debug("final layer p1", Double.toString(p1));
                            //Log.debug("final layer p2", Double.toString(p2));

                            double p3 = model[layer - 1][pL][1];

                            //Log.debug("final layer p3", Double.toString(p3));

                            double delta = p1 * p2 * p3;
                            double newWeight = weights[layer - 1][x][pL] - this.learningRate * delta;

                            //Log.debug("final layer delta learningRate", Double.toString(learningRate));
                            //Log.debug("final layer delta output", Double.toString(delta));
                            //Log.debug("final layer old weight output", Double.toString(weights[layer - 1][x][pL]));

                            weightsNew[layer - 1][x][pL] = newWeight;

                            //Log.debug("final layer new weight output", Double.toString(weightsNew[layer - 1][x][pL]));
                        }

                        // save for later usage further down the backpropagation process
                        model[layer][x][2] = p1;
                        model[layer][x][3] = p2;
                    }

                    //System.exit(0);


                }else{

                    for(int x = 0; x <  model[layer].length; x++) {

                        double out = model[layer][x][1];
                        double p2 = this.logisticDerivative(out);
                        double p1tot = 0;


                        // worked initially here
                        //double p1a = model[layer + 1][x][2] * model[layer + 1][x][3];



                        // taking care of the spread of the output according to weights
                        for(int pL = 0; pL <  model[layer + 1].length; pL++) {



                            double p1a = model[layer + 1][pL][2] * model[layer + 1][pL][3];

                            //Log.debug("p1a", Double.toString(p1a));

                            // this is maybe wrong mixing up x and pL
                            //double w = weights[layer][x][pL];
                            double w = weights[layer][pL][x];

                            p1tot += p1a * w;
                        }



                        // now loop through all neuron of previous layer
                        for(int pL = 0; pL <  model[layer - 1].length; pL++) {

                            double p3 = model[layer - 1][pL][1];

                            //Log.debug("p1tot", Double.toString(p1tot));

                            //Log.debug("p2", Double.toString(p2));

                            //Log.debug("p3", Double.toString(p3));

                            double delta = p1tot * p2 * p3;
                            //Log.debug("delta learningRate", Double.toString(learningRate));
                            //Log.debug("delta hidden", Double.toString(delta));

                            double newWeight = weights[layer - 1][x][pL] - this.learningRate * delta;

                            //Log.debug("old weight hidden", Double.toString(weights[layer - 1][x][pL]));

                            weightsNew[layer - 1][x][pL] = newWeight;

                            //Log.debug("new weight hidden", Double.toString(weightsNew[layer - 1][x][pL]));
                        }


                        // save for later usage
                        model[layer][x][2] = p1tot;
                        model[layer][x][3] = p2;

                    }
                }
            }

            weights = weightsNew;


        } // next loop for same image

        // make one hot encoding of final layer
        //debugResults();


    }


    private void debugResults(){

        double one[] = new double[10];

        for(int t = 0; t < 10; t++){
            one[t] = model[model.length - 1][t][1];
        }

        String statement = " Error initial: " + Double.toString(initialError) + " Error final: " + Double.toString(totalError);
        statement += " learning rate now: " + Double.toString(learningRate);
        statement += " category: " + currentCategory;
        statement += " prediction: " + oneHot(one);
        Log.important("Result: ", statement);


    }

    private String oneHot(double[] arr){

        int ret = 0;
        double tst = 0;
        String a = "";

        for(int i = 0; i < arr.length; i++){

            if(arr[i] > tst){
                tst = arr[i];
                ret = i;
            }

            a += arr[i] + "|";
        }
        return Integer.toString(ret) + " Arr: " + a;
    }

    private double logisticDerivative(double x){
        return x * (1 - x);
    }


    public double logisticFunction(double i){

        // https://en.wikipedia.org/wiki/Logistic_function

        // https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
        double r;
        double x = 1 / (1 + Math.pow(Math.E,  -i));
        r = x;
        return r;
    }

}
