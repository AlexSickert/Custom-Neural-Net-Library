package datayield;

import java.util.ArrayList;

public class MainArraySimple {

    public static void main(String[] args) {
	// write your code here

        Log.log("info", "start");

        double[][] x = new double[10][4];

        x[0] = new double[]{0,0,0,0};
        x[1] = new double[]{0,0,0,1};
        x[2] = new double[]{0,0,1,0};
        x[3] = new double[]{0,0,1,1};
        x[4] = new double[]{0,1,0,0};
        x[5] = new double[]{0,1,0,1};
        x[6] = new double[]{0,1,1,0};
        x[7] = new double[]{0,1,1,1};
        x[8] = new double[]{1,0,0,0};
        x[9] = new double[]{1,0,0,1};

        double[][] xStupid = new double[4][4];

        xStupid[0] = new double[]{0,1,0,0};
        xStupid[1] = new double[]{0,1,0,0};
        xStupid[2] = new double[]{0,0,1,0};
        xStupid[3] = new double[]{0,0,1,0};

        double[][] y = new double[10][10];

        y[0] = new double[]{1,0,0,0,0,0,0,0,0,0};
        y[1] = new double[]{0,1,0,0,0,0,0,0,0,0};
        y[2] = new double[]{0,0,1,0,0,0,0,0,0,0};
        y[3] = new double[]{0,0,0,1,0,0,0,0,0,0};
        y[4] = new double[]{0,0,0,0,1,0,0,0,0,0};
        y[5] = new double[]{0,0,0,0,0,1,0,0,0,0};
        y[6] = new double[]{0,0,0,0,0,0,1,0,0,0};
        y[7] = new double[]{0,0,0,0,0,0,0,1,0,0};
        y[8] = new double[]{0,0,0,0,0,0,0,0,1,0};
        y[9] = new double[]{0,0,0,0,0,0,0,0,0,1};

        double[][] yStupid = new double[4][10];

        yStupid[0] = new double[]{0,0,0,0,0,0,0,0,1,0};
        yStupid[1] = new double[]{0,0,0,0,0,0,0,0,1,0};
        yStupid[2] = new double[]{0,1,0,0,0,0,0,0,0,0};
        yStupid[3] = new double[]{0,1,0,0,0,0,0,0,0,0};

        DyNnArraySimple nn2 = new DyNnArraySimple();
        //nn2.train();

        nn2.addInputLayer(4);
        nn2.addHiddenLayer(4);
        //nn2.addHiddenLayer(4);
        nn2.addHiddenLayer(10);
        nn2.train(y, x);
        //nn2.train(yStupid, xStupid);

    }
}
