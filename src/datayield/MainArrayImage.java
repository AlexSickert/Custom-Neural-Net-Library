package datayield;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Random;

public class MainArrayImage {

    public static void main(String[] args) {
	// write your code here

        Log.log("info", "start");

//        double[][] x = new double[10][4];
//
//        x[0] = new double[]{0,0,0,0};
//        x[1] = new double[]{0,0,0,1};
//        x[2] = new double[]{0,0,1,0};
//        x[3] = new double[]{0,0,1,1};
//        x[4] = new double[]{0,1,0,0};
//        x[5] = new double[]{0,1,0,1};
//        x[6] = new double[]{0,1,1,0};
//        x[7] = new double[]{0,1,1,1};
//        x[8] = new double[]{1,0,0,0};
//        x[9] = new double[]{1,0,0,1};
//
//        double[][] xStupid = new double[4][4];
//
//        xStupid[0] = new double[]{0,1,0,0};
//        xStupid[1] = new double[]{0,1,0,0};
//        xStupid[2] = new double[]{0,0,1,0};
//        xStupid[3] = new double[]{0,0,1,0};
//
//        double[][] y = new double[10][10];
//
//        y[0] = new double[]{1,0,0,0,0,0,0,0,0,0};
//        y[1] = new double[]{0,1,0,0,0,0,0,0,0,0};
//        y[2] = new double[]{0,0,1,0,0,0,0,0,0,0};
//        y[3] = new double[]{0,0,0,1,0,0,0,0,0,0};
//        y[4] = new double[]{0,0,0,0,1,0,0,0,0,0};
//        y[5] = new double[]{0,0,0,0,0,1,0,0,0,0};
//        y[6] = new double[]{0,0,0,0,0,0,1,0,0,0};
//        y[7] = new double[]{0,0,0,0,0,0,0,1,0,0};
//        y[8] = new double[]{0,0,0,0,0,0,0,0,1,0};
//        y[9] = new double[]{0,0,0,0,0,0,0,0,0,1};
//
//        double[][] yStupid = new double[4][10];
//
//        yStupid[0] = new double[]{0,0,0,0,0,0,0,0,1,0};
//        yStupid[1] = new double[]{0,0,0,0,0,0,0,0,1,0};
//        yStupid[2] = new double[]{0,1,0,0,0,0,0,0,0,0};
//        yStupid[3] = new double[]{0,1,0,0,0,0,0,0,0,0};
//
//        DyNnArraySimple nn2 = new DyNnArraySimple();
//        //nn2.train();
//
//        nn2.addInputLayer(4);
//        nn2.addHiddenLayer(4);
//        //nn2.addHiddenLayer(4);
//        nn2.addHiddenLayer(10);
//        nn2.train(y, x);
//        //nn2.train(yStupid, xStupid);



        String basePath = "C:\\Users\\alex\\IdeaProjects\\DyNeuralNet\\Data\\training\\";

        double[][] x = new double[5000][784];
        double[][] y = new double[5000][10];

        System.out.println("making image array");

        for(int k = 0; k < 5000; k++){

            //System.out.println("-------------");

            String f = Long.toString(Math.round(Math.random() * 9));
            //System.out.println(f);
            String randomPath = getRandomPath(basePath, f);
            //Log.log("info", randomPath);
            double[] inputArr = getArrayFromImage(randomPath);
            x[k] = inputArr;
            y[k] = getTarget(f);

        }

        DyNnArrayImage nn2 = new DyNnArrayImage();
        //nn2.train();

        nn2.addInputLayer(784);
        nn2.addHiddenLayer(700);
        nn2.addHiddenLayer(600);
        nn2.addHiddenLayer(500);
        nn2.addHiddenLayer(400);
        nn2.addHiddenLayer(300);
        nn2.addHiddenLayer(200);
        nn2.addHiddenLayer(100);
        //nn2.addHiddenLayer(4);
        nn2.addHiddenLayer(10);

        System.out.println("start training");

        nn2.setHyperParameters(1000, 3, 300, 0.1, 0.99);

        nn2.train(y, x);
        //nn2.train(yStupid, xStupid);


    }

    private static double[] getTarget(String f){

        double[] ret = {0,0,0,0,0,0,0,0,0,0};
        int p = Integer.decode(f);
        ret[p] = 1;
        return ret;
    }


    private static String getRandomPath(String basePath, String f){

        try {

            basePath += f;

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


    public static double[] getArrayFromImage (String ImageName) {

        double[] data;

        try {

            BufferedImage bImage = ImageIO.read(new File(ImageName));

            //System.out.println(bImage.getHeight());
            int x = bImage.getHeight();
            //System.out.println(bImage.getWidth());
            int y = bImage.getWidth();
            double xy;
            data = new double[y * x];
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
}
