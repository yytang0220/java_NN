import java.util.ArrayList;
import java.util.Random;

public class checking_test {

    public static void main(String[] args) {
        ArrayList<Integer[]> testdata = new ArrayList<>();
        Integer[] line;
        Random cool = new Random(12193);
        for (int jj = 0; jj < 5; jj++) {
            line = new Integer[3];
            line[0] = 12;
            line[1] = 12;
            line[2] = 12;
//            for (int ii = 0; ii < 3; ii++) {
//                line[ii] =
//            }
            testdata.add(line);
        }
        checking checker = new checking(2);
        if (checker.check_sentence(testdata)) System.out.println("yes");
        else System.out.println("no");
    }
}
