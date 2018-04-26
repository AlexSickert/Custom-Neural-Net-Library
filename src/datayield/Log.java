package datayield;

public class Log {
    public static void log( String topic, String text){
        String s = topic + " : " + text;
        System.out.println(s);
    }

    public static void debug(String topic, String text){
        String s = topic + " : " + text;
        System.out.println(s);
    }

    public static void error(String topic, String text){
        String s = topic + " : " + text;
        System.out.println(s);
    }

    public static void important(String topic, String text){
        String s = topic + " : " + text;
        System.out.println(s);
    }
}
