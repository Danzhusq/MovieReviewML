/**
 * Created by danny on 4/27/2017.
 */
public abstract class MachineLearningModel {
    protected int tPositive;
    protected int fPositive;
    protected int tNegative;
    protected int fNegative;

    protected double accuracy;
    protected double recall;
    protected double precision;

    protected abstract void train();

    protected abstract void test();

    /**
     * Calcultate performance of accuracy, precision and recall using provided test data.
     */
    protected void calculatePerformance() {
        double precisionPos = (tPositive / (double)(tPositive + fPositive));
        double precisionNeg = (tNegative / (double)(tNegative + fNegative));
        precision = ( (precisionNeg + precisionPos) / 2.0);
        accuracy = ( ((tPositive + tNegative) / (double)(tPositive + fPositive + tNegative + fNegative)));
        double recallPos = ( (tPositive) / (double) (tPositive + fNegative));
        double recallNeg = ( (tNegative) / (double)( tNegative + fPositive));
        recall = (recallNeg + recallPos) / 2.0;
    }

    @Override
    /**
     * Override toString method to provided standart stats report.
     */
    public String toString() {
        return "Number of True Positive: " + tPositive
                + "\n Number of False Positive: " + fPositive
                + "\n Number of True Negative: " + tNegative
                + "\n Number of False Negative: " + fNegative
                + "\n Accuracy: " + accuracy
                + "\n Recall: " + recall
                + "\n Precision: " + precision;
    }
}
