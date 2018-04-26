package datayield;

import java.util.ArrayList;
import java.util.Arrays;


public class DyNn {

    private float model[][][] = new float[3][2][4];
    private float[][][] weights  = new float[2][2][2];
    private float[][][] weightsNew  = new float[2][2][2];;
    private float[] errors;
    private float[] target;
    private float[] inputs;
    private int loops;
    private float learningRate;


    /*

    y values of the model:

    0 = values
    1 = nets
    2 = outputs
    3 = type of node


     */


    public void setInputLayer(int x){

        //model[0] = makeLayer(x);

        // for each input node we need several
        //model.add(makeLayer(x));
    }

    private ArrayList makeLayer(int x){

        ArrayList xVal = new ArrayList();

        for(int i = 0; i < x; i++){
            int[] yVal = new int[10];
            xVal.add(new ArrayList<>(Arrays.asList(yVal)));
        }
        return xVal;
    }

    public void setOutputLayer(int x){
        //model = appendValue(model, makeLayer(x));
        //model.add(makeLayer(x));
    }

    public void addLayer(int x){
        //model = appendValue(model, makeLayer(x));
        //model.add(makeLayer(x));
    }

    public void setHyperParameters(int loops, float learningRate){
        this.loops = loops;
        this.learningRate = learningRate;
    }

    public void setInputData(ArrayList<Float> arr){

        this.inputs =  new float[arr.size()];

        for(int i = 0; i < arr.size(); i++){
            this.inputs[i] = arr.get(i);
        }
    }

    public void setTargetData(ArrayList<Float> arr){

        this.target =  new float[arr.size()];

        for(int i = 0; i < arr.size(); i++){
            this.target[i] = arr.get(i);
        }
    }

    public void train(){

        // fill input data to the model
        // convert ArrayList to Array

        Log.log("next step", "--- fill input data ---");

        for(int i = 0; i < this.inputs.length; i++){
            model[0][i][1] = this.inputs[i];
        }

        // create the weights array
        int layerBeforeLength = 0;
        int layerNowLength = 0;


        Log.log("next step", "--- initialize weights ---");

        for(int i = 0; i < this.model.length; i++){

            Log.log("i", Integer.toString(i));

            if(i > 0){
                layerNowLength =  model[i].length;

                for(int x = 0; x < layerNowLength; x ++){

                    Log.log("x", Integer.toString(x));


                    for(int xBefore = 0; xBefore < layerBeforeLength; xBefore ++){

                        Log.log("xBefore", Integer.toString(xBefore));

                        weights[i-1][x][xBefore] = (float) Math.random();
                    }
                }
            }
            layerBeforeLength = model[i].length;
        }

        // now start train loop
        Log.log("next step", "--- now starting forward calculation ---");
        for(int i = 0; i < loops; i++){

            forwardCalculation();
            backwardCalculation();
            updateWeights();

        }

    }

    private void updateWeights(){
        Log.log("next step", "--- now updating weights ---");

        weights = this.weightsNew;


    }

    private float deltaRule(float target, float out, float outHidden){
        float ret;
        ret = -(target - out) * out * (1-out) * outHidden;
        return ret;
    }

    private void backwardCalculation(){

        Log.log("next step", "--- now starting backward calculation ---");

        for(int i = this.model.length -1 ; i >0; i--){

            if(i == this.model.length -1){

                // last layer special calculation

                // for each input x we calculate the new weight
                //  model[0][i][1]
                //  weights[i-1][x][xBefore]

                int layerLength = model[i].length;

                for(int xOut = 0; xOut < layerLength; xOut++){

                    float targetVal = target[xOut];
                    float outVal = model[i][xOut][1];

                    int xInLen = model[i-1].length;

                    for(int xIn = 0; xIn < xInLen; xIn++){
                        Log.log("next output neuron", "-----------------------------------");
                        float outHidden = model[i-1][xIn][1]; // the output value of the last hidden layer

                        float val = deltaRule(targetVal, outVal, outHidden);
                        float newWeight;

                        Log.log("targetVal", Float.toString(targetVal));
                        Log.log("outVal", Float.toString(outVal));
                        Log.log("outHidden", Float.toString(outHidden));
                        Log.log("val", Float.toString(val));

                        newWeight = weights[i-1][xOut][xIn] * learningRate * val;

                        Log.log("newWeight", Float.toString(newWeight));

                        weightsNew[i-1][xOut][xIn] = newWeight;

                    }
                }

            }else{

                // processing hidden layers

                // for each X in this hidden layer
                int layerLength = model[i].length;

                for(int xHidden = 0; xHidden < layerLength; xHidden++){

                    // create sum of output layer impact

                    float v = 0;

                    // summarize impact for each output neuron
                    for(int xOut = 0; xOut < target.length; xOut++){

                        float t = target[xOut];
                        float o = model[this.model.length -1][xOut][1];

                        // the outgoing weights
                        float w = weights[i][xOut][xHidden];

                        v += -(t - o) *  o * (1 - o) * w;
                    }

                    // now we have all the sums
                    float partTwo;
                    float outH;

                    outH = model[i][xHidden][1];
                    partTwo = outH * (1 - outH);

                    // now loop over all input weights
                    int layerBeforeLength = model[i-1].length;

                    for(int z = 0; z < layerBeforeLength; z++){

                        float outBefore = model[i-1][z][1];
                        float weightBefore = weights[i][xHidden][z];

                        float delta;
                        float newWeight;

                        delta = v * partTwo * outBefore;

                        newWeight = weightBefore - learningRate * delta;
                        weightsNew[i][xHidden][z] = newWeight;
                    }
                }
            }
        }
    }

    private void forwardCalculation(){

        int layerBeforeLength = 0;
        int layerNowLength = 0;

        // check input values in first layer


        Log.log("input layer length", Integer.toString(this.model[0].length));

        for(int i = 0; i < this.model[0].length; i++){
            Log.log("input layer value", Float.toString(model[0][i][1]));
        }


        for(int i = 0; i < this.model.length; i++){

            if(i > 0){
                layerNowLength =  model[i].length;

                for(int x = 0; x < layerNowLength; x ++){

                    float netVal = 0;

                    for(int xBefore = 0; xBefore < layerBeforeLength; xBefore ++){

                        Log.log("node input value", Float.toString(model[i-1][xBefore][1]));
                        Log.log("weight value", Float.toString(weights[i-1][x][xBefore]));

                        netVal += model[i-1][xBefore][1] * weights[i-1][x][xBefore];
                    }

                    // set the value
                    model[i][x][0] = netVal;
                    // apply logistic function to calculate output value
                    model[i][x][1] = logisticFunction(netVal);

                }
            }
            layerBeforeLength = model[i].length;
        }

        // calculate errors
        int outputLayerIndex = this.model.length -1;

        layerNowLength =  model[outputLayerIndex].length;

        float totalError = 0;

        for(int x = 0; x < layerNowLength; x ++){

            float output = model[outputLayerIndex][x][0];
            Log.log("output", Float.toString(output));
            float target = this.target[x];
            Log.log("target", Float.toString(target));
            float diff = target - output;
            Log.log("diff", Float.toString(diff));
            float error = 0.5F * (float) Math.pow((float) diff, 2);
            totalError += error;
        }

        Log.debug("Total error:", Float.toString(totalError));
    }




    public float logisticFunction(float i){

        // https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
        float r;
        double x = 1 / (1 + Math.pow(Math.E,  -i));
        r = (float) x;
        return r;
    }

}
