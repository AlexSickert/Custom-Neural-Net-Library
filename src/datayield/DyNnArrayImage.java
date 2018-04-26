package datayield;

import java.util.ArrayList;


public class DyNnArrayImage {

    private double model[][][];
    private double[][][] weights;
    //private double[][][] weightsNew;

    private String currentCategory = "";

    ArrayList modelConstructor = new ArrayList();

    private double errors[];
    //private double initialError = 0;
    private double totalError = 0;

    private double target[];
    private double inputs[];
    private int epochs = 100;
    private int loopsPerImage = 30;
    private int samples = 100;
    private double learningRate = 0.0001;
    private double decay = 0.9999;

    /*

    y values of the model:

    0 = net
    1 = output
    2 = saved value for next backpropagation step
    3 = saved value for next backpropagation step

     */


    public void setHyperParameters(int epochs, int loopsPerImage, int samples, double learningRate, double decay){
        this.epochs = epochs;
        this.loopsPerImage = loopsPerImage;
        this.samples = samples;
        this.learningRate = learningRate;
        this.decay = decay;

    }

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
        //weightsNew = new double[s-1][][];

        int xBefore = 0;

        for(int i = 0; i < s; i++){
            int x = (int) modelConstructor.get(i);
            double[][] table = new double[x][5];
            model[i] = table;

            if(i > 0){
                double[][] wT = new double[x][xBefore];
                weights[i-1] = wT;
                double[][] wTN = new double[x][xBefore];
                //weightsNew[i-1] = wTN;
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


    /**
     * trains the model using the dataset consisting of x and y
     * @param y
     * @param x
     */
    public void train(double[][] y, double[][] x){

        constructModel();
        initializeWeights();

        //target

        for(int e = 0; e < epochs; e++){

            System.out.println("epoch: " + e);

            for(int i = 0; i < samples; i++){

                // choose random sample from total data set
                int k = (int) Math.round(Math.random() * (y.length -1));
                //System.out.println(k);
                double[] yOne = y[k];
                double[] xOne = x[k];
                target = yOne;
                trainOneSample(xOne);
            }

            learningRate = learningRate * decay;

            System.out.println("new learning rate: " + learningRate);

            calculateAccuracy(y, x);
        }
    }

    /**
     * Runs a few predictions and counts the correct predictions and out of that calculates the accuracy
     * @param y
     * @param x
     */
    private void calculateAccuracy(double[][] y, double[][] x){

        // make a few tests

        double countOK = 0;
        double countAll = 0;
        double ratio;

        for(int z = 0; z < 100; z++){

            int k = (int) Math.round(Math.random() * (y.length -1));
            double[] yOne = y[k];
            double[] xOne = x[k];

            if(predictOneSampleWithCheck(yOne, xOne)){
                countOK += 1;
            }
            countAll += 1;
        }

        ratio = countOK / countAll;
        ratio = ratio * 100;

        System.out.println("Accuracy in %: " + ratio);
    }


    /**
     * Fill the weights array with the initial random values
     */
    private void initializeWeights(){

        // initialize weights
        for(int c = 0; c < weights.length; c++){
            for(int y = 0; y < weights[c].length; y++) {
                for(int z = 0; z < weights[c][y].length; z++) {
                    //weights[c][y][z] = Math.random() * 0.1;
                    weights[c][y][z] = Math.random() - 0.5;
                    //weights[c][y][z] = 0.00001;
                }
            }
        }
    }

    /**
     * Fills the model with the input values
     * @param inputArr
     */
    private void setInputVariables(double inputArr[]){
        // set input values
        for(int k = 0; k < model[0].length; k++){
            model[0][k][1] = inputArr[k];
            //Log.debug("input", Double.toString( model[0][k][1]));
        }
    }


    /**
     * Performs a prediction of one sample and returns if the prediction was accurate or not.
     * @param arrTarget
     * @param arr
     * @return if prediction was correct or not.
     */
    private boolean predictOneSampleWithCheck(double[] arrTarget, double[] arr){

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

        // get values from last layer
        for(int t = 0; t < 10; t++){
            one[t] = model[model.length - 1][t][1];
        }

//        String statement = "===> ";
//        statement += " totalError: " +  totalError;
//        statement += " target: " +  oneHotText(arrTarget);
//        statement += " prediction: " + oneHotText(one);
//        System.out.println(statement);

        if(oneHotInt(arrTarget) == oneHotInt(one)){
            return true;
        }else{
            return false;
        }
    }


    /**
     * Trains one input sample and trains with it.
     * @param inputArr
     */
    private void trainOneSample(double inputArr[]){



        setInputVariables(inputArr);


        // do all training loopsPerImage
        for(int i = 0; i < loopsPerImage; i++){

            forwardPass();
            calculateError();
            backwardPropagation();
            updateWeights();

        } // next loop for same image

        // make one hot encoding of final layer
        //debugResults();


    }

    /**
     * takes the input values and calculates the output using the weights of the weights array.
     */
    private void forwardPass(){

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

                    model[layer][x][0] = net;
                    //Log.debug("net value", Double.toString(model[layer][x][0]));
                    model[layer][x][1] = logisticFunction(net);
                    //Log.debug("output value", Double.toString(model[layer][x][1]));
                }
            }
        }
    }

    /**
     * calcualtes the prediction error across all output neurons
     */
    private void calculateError(){


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




    }

    /**
     * perform the backward propagation
     */
    private void backwardPropagation(){

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

                        //weightsNew[layer - 1][x][pL] = newWeight;
                        updateOneWeight(layer - 1, x, pL, newWeight);

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

                        //weightsNew[layer - 1][x][pL] = newWeight;
                        updateOneWeight(layer - 1, x, pL, newWeight);

                        //System.out.println(layer);
                        if(layer == 1 && x == 3 && pL == 3){
                            //System.out.println(newWeight);
                        }

                        //Log.debug("new weight hidden", Double.toString(weightsNew[layer - 1][x][pL]));
                    }

                    // save for later usage
                    model[layer][x][2] = p1tot;
                    model[layer][x][3] = p2;
                }
            }
        }
    }

    /**
     * Update the weights array
     * @param x
     * @param y
     * @param z
     * @param val
     */
    private void updateOneWeight(int x, int y, int z, double val){
        //weightsNew[x][y][z] = val;
        weights[x][y][z] = val;
    }

    /**
     * it seems that this method is obsolete and can be removed
     */
    private void updateWeights(){
        //weights = weightsNew;
    }

    private void debugResults(){

        double one[] = new double[10];

        for(int t = 0; t < 10; t++){
            one[t] = model[model.length - 1][t][1];
        }

        String statement = " Error final: " + Double.toString(totalError);
        statement += " learning rate now: " + Double.toString(learningRate);
        statement += " category: " + currentCategory;
        statement += " prediction: " + oneHotText(one);
        Log.important("Result: ", statement);

    }

    /**
     * Helper to create one hot encoding. This is only for debugging purposes.
     * @see Method that returns desired interger value is oneHotInt()
     * @param arr Array with numeric values
     * @return the index of the array element with the highest value
     */
    private String oneHotText(double[] arr){

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

    /**
     * Return the index of the array element with the highest value
     * @param arr
     * @return
     */
    private int oneHotInt(double[] arr){

        int ret = 0;
        double tst = 0;

        for(int i = 0; i < arr.length; i++){

            if(arr[i] > tst){
                tst = arr[i];
                ret = i;
            }
        }
        return ret;
    }

    /**
     *
     * @param x
     * @return
     */
    private double logisticDerivative(double x){
        return x * (1 - x);
    }


    /**
     * Logisitc function to convert the net inputs of the incoming neurons to the output value
     * https://en.wikipedia.org/wiki/Logistic_function
     * https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
     * @param i Net input, sum of all input neurons multiplied by weight
     * @return value between 0 and 1
     */
    private double logisticFunction(double i){

        double r;
        double x = 1 / (1 + Math.pow(Math.E,  -i));
        r = x;
        return r;
    }

}
