package datayield;

public class DyNnSingle {

    private double model[][] = new double[5][5];
    private double[] weights  = new double[4];
    private double[] weightsNew  = new double[4];;
    private double errors = 0;
    private double initialError = 0;

    private double target = 0.8;
    private double inputs  = 0.2;
    private int loops;
    private double learningRate = 0.01;

    /*

    y values of the model:

    0 = values
    1 = nets
    2 = outputs
    3 = type of node

     */

    public void run(){

        loops = 100000;

        // initialize weights
        for(int i = 0; i < weights.length; i++){
            weights[i] = Math.random();
        }

        // set input values
        model[0][0] = inputs;


        for(int i = 0; i < loops; i++){

            Log.debug("-------------", "-----------------------------");

            // forward pass
            for(int layer = 0; layer < model.length; layer++){

                if(layer > 0){
                    double net = weights[layer - 1] * model[layer-1][0];
                    model[layer][0] = net;
                    Log.debug("net value", Double.toString(model[layer][0]));
                    model[layer][1] = this.logisticFunction(net);
                    Log.debug("output value", Double.toString(model[layer][1]));
                }
            }

            // calculate error
            errors = target - model[model.length-1][1];
            errors = 0.5 * Math.pow(errors, 2);

            Log.important("Error now", Double.toString(errors));

            if(i == 0){
                initialError = errors;
            }


            // backwards
            for(int layer = model.length-1; layer > 0; layer--){

                // special case last layer
                if(layer == model.length-1){

                    double out = model[model.length-1][1];
                    double p1 = out - target;
                    double p2 = this.logisticDerivative(out);
                    double p3 = model[model.length-2][1];
                    double delta = p1 * p2 * p3;
                    double newWeight = weights[layer - 1] - this.learningRate * delta;

                    Log.debug("delta learningRate", Double.toString(learningRate));
                    Log.debug("delta output", Double.toString(delta));
                    Log.debug("old weight output", Double.toString(weights[layer - 1]));

                    weightsNew[layer - 1] = newWeight;

                    Log.debug("new weight output", Double.toString(weightsNew[layer - 1]));

                    // save for later usage
                    model[layer][2] = p1;
                    model[layer][3] = p2;


                }else{
                    double out = model[layer][1];
                    // here we use the saved values from the laer before
                    double p1a = model[layer + 1][2] * model[layer + 1][3];
                    double p1 = p1a * weights[layer - 1];
                    double p2 = this.logisticDerivative(out);
                    double p3 = model[layer - 1][1];

                    double delta = p1 * p2 * p3;

                    Log.debug("delta learningRate", Double.toString(learningRate));
                    Log.debug("delta hidden", Double.toString(delta));

                    double newWeight = weights[layer - 1] - this.learningRate * delta;

                    Log.debug("old weight hidden", Double.toString(weights[layer - 1]));

                    weightsNew[layer - 1] = newWeight;

                    Log.debug("new weight hidden", Double.toString(weightsNew[layer - 1]));

                    // save for later usage
                    model[layer][2] = p1;
                    model[layer][3] = p2;
                }
            }


            // copy weights
            for(int z = 0; z < weights.length; z++){
                weights[z] = new Double(weightsNew[z]);
            }
        }

        Log.important("Error final", Double.toString(errors));
        Log.important("Error initial", Double.toString(initialError));

    }


    private double logisticDerivative(double x){
        return x * (1 - x);
    }


    public double logisticFunction(double i){

        // https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
        double r;
        double x = 1 / (1 + Math.pow(Math.E,  -i));
        r = x;
        return r;
    }

}
