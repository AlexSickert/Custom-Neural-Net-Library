package datayield;

import java.util.ArrayList;

public class Main {

    public static void main(String[] args) {
	// write your code here

        Log.log("info", "start");

        DyNn nn = new DyNn();

        nn.setInputLayer( 2);
        nn.addLayer(2);
        nn.setOutputLayer(2);

        nn.setHyperParameters(10, 0.0001F);

        ArrayList<Float> arl = new ArrayList();
        arl.add(0.99F);
        arl.add(0.1F);
        nn.setInputData(arl);

        ArrayList<Float> arlO = new ArrayList();
        arlO.add(0.99F);
        arlO.add(0.1F);

        nn.setTargetData(arlO);

        //nn.train();

        //System.out.println(nn.logisticFunction(-1.000F));

        //DyNnSingle nn2 = new DyNnSingle();
        //nn2.run();

        DyNnArray nn2 = new DyNnArray();
        //nn2.train();

        nn2.addInputLayer(784);
        nn2.addHiddenLayer(500);
        nn2.addHiddenLayer(400);
        nn2.addHiddenLayer(300);
        nn2.addHiddenLayer(200);
        nn2.addHiddenLayer(100);
        nn2.addHiddenLayer(10);
        nn2.train();

        //nn2.testImage();


    }
}
