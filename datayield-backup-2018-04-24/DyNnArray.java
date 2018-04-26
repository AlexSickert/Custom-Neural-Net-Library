package datayield;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.util.Random;
import java.nio.file.Files;
import java.io.File;


public class DyNnArray {

    private double model[][][] = new double[5][5][5];
    private double[][][] weights  = new double[4][5][5];
    private double[][][] weightsNew  = new double[4][5][5];;
    private double errors[] = new double[5];
    private double initialError = 0;
    private double totalError = 0;

    private double target[] = {0.1,0.2,0.3,0.4,0.5};
    private double inputs[]  = {0.9,0.8,0.7,0.6,0.5};
    private int loops = 10;
    private int samples = 10;
    private double learningRate = 0.01;

    /*

    y values of the model:

    0 = net
    1 = output
    2 = saved value for next backpropagation step
    3 = saved value for next backpropagation step

     */

    private String getRandomPath(String basePath, String f){

        basePath += f;
        Log.error("base path", basePath);

        File dir = new File(basePath);
        File[] files = dir.listFiles();
        Random rand = new Random();
        File file = files[rand.nextInt(files.length)];
        String r = file.getPath();
        return r;
    }


    public byte[] extractBytes (String ImageName) throws IOException {

        File imgPath = new File(ImageName);
        BufferedImage bufferedImage = ImageIO.read(imgPath);

        // get DataBufferBytes from Raster
        WritableRaster raster = bufferedImage.getRaster();
        DataBufferByte data   = (DataBufferByte) raster.getDataBuffer();

        System.out.println(data.getData().length);

        return ( data.getData() );

    }

    private double[] getInput(String path){

        double retArr[];

        try{
            byte[] fileContent = extractBytes (path);

            retArr = new double[fileContent.length];


            for(int i = 0; i < fileContent.length; i++){

                //System.out.println(fileContent[i]);

                Byte b = new Byte(fileContent[i]);
                int x = b.intValue();
                x = x + 128;

                if (x  < 0){
                    System.out.println(x);
                }

                if (x  > 256){
                    System.out.println(x);
                }

                double d = (double) x;

                d = d/ 256;
                //System.out.println(d);
                retArr[i] =  d;
            }

            return retArr;

        }catch(Exception x){

            Log.error("getInput()", x.toString());

        }

        return null;

    }
    private double[] getTarget(String f){

        double[] ret = {0,0,0,0,0,0,0,0,0,0};



        int p = Integer.decode(f);

        ret[p] = 1;

        return ret;
    }

    public void train(){

        initializeWeights();

        //target

        for(int i = 0; i < samples; i++){

            String basePath = "C:\\Users\\alex\\IdeaProjects\\DyNeuralNet\\Data\\training\\";

            String f = Long.toString(Math.round(Math.random() * 10));

            String randomPath = getRandomPath(basePath, f);

            double inputArr[] = getInput(randomPath);
            double target[] = getTarget(f);




            //trainOneSample(inputArr);


        }


    }





    private void initializeWeights(){


        // initialize weights
        for(int c = 0; c < weights.length; c++){
            for(int y = 0; y < weights[c].length; y++) {
                for(int z = 0; z < weights[c][y].length; z++) {
                    weights[c][y][z] = Math.random();
                }
            }
        }
    }

    private void setInputVariables(double inputArr[]){

        // set input values
        for(int k = 0; k < model[0].length; k++){
            model[0][k][1] = inputs[k];
            Log.debug("input", Double.toString( model[0][k][1]));
        }

    }


    private void trainOneSample(double inputArr[]){



        setInputVariables(inputArr);


        // do all training loops
        for(int i = 0; i < loops; i++){


            Log.debug("-------------", "-----------------------------");

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
                            net += weights[layer - 1][x][n] * tst;
                        }

                        model[layer][x][0] = net;
                        Log.debug("net value", Double.toString(model[layer][x][0]));
                        model[layer][x][1] = logisticFunction(net);
                        Log.debug("output value", Double.toString(model[layer][x][1]));
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
                        double p1 = out - target[x];
                        double p2 = this.logisticDerivative(out);

                        Log.debug("p1", Double.toString(p1));
                        Log.debug("p2", Double.toString(p2));

                        // now loop through all neuron of previous layer
                        for(int pL = 0; pL <  model[layer - 1].length; pL++) {

                            double p3 = model[layer - 1][pL][1];

                            Log.debug("p3", Double.toString(p3));

                            double delta = p1 * p2 * p3;
                            double newWeight = weights[layer - 1][x][pL] - this.learningRate * delta;

                            Log.debug("delta learningRate", Double.toString(learningRate));
                            Log.debug("delta output", Double.toString(delta));
                            Log.debug("old weight output", Double.toString(weights[layer - 1][x][pL]));

                            weightsNew[layer - 1][x][pL] = newWeight;

                            Log.debug("new weight output", Double.toString(weightsNew[layer - 1][x][pL]));
                        }

                        // save for later usage further down the backpropagation process
                        model[layer][x][2] = p1;
                        model[layer][x][3] = p2;
                    }
                }else{

                    for(int x = 0; x <  model[layer].length; x++) {

                        double out = model[layer][x][1];
                        double p2 = this.logisticDerivative(out);
                        double p1tot = 0;

                        Log.debug("p2", Double.toString(p2));

                        double p1a = model[layer + 1][x][2] * model[layer + 1][x][3];

                        Log.debug("p1a", Double.toString(p1a));

                        // taking care of the spread of the output according to weights
                        for(int pL = 0; pL <  model[layer + 1].length; pL++) {
                            // this is maybe wrong mixing up x and pL
                            double w = weights[layer][x][pL];
                            p1tot += p1a * w;
                        }

                        Log.debug("p1tot", Double.toString(p1tot));

                        // now loop through all neuron of previous layer
                        for(int pL = 0; pL <  model[layer - 1].length; pL++) {

                            double p3 = model[layer - 1][pL][1];

                            Log.debug("p3", Double.toString(p3));

                            double delta = p1tot * p2 * p3;
                            Log.debug("delta learningRate", Double.toString(learningRate));
                            Log.debug("delta hidden", Double.toString(delta));

                            double newWeight = weights[layer - 1][x][pL] - this.learningRate * delta;

                            Log.debug("old weight hidden", Double.toString(weights[layer - 1][x][pL]));

                            weightsNew[layer - 1][x][pL] = newWeight;

                            Log.debug("new weight hidden", Double.toString(weightsNew[layer - 1][x][pL]));
                        }


                        // save for later usage
                        model[layer][x][2] = p1tot;
                        model[layer][x][3] = p2;

                    }
                }
            }

            weights = weightsNew;
        }

        Log.important("Error initial", Double.toString(initialError));
        Log.important("Error final", Double.toString(totalError));


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
