package com.javadeeplearningcookbook.examples;

import java.math.BigDecimal;
import java.util.Formatter;

public class AssessmentMetrics {
    private Integer intTruePositive;
    private Integer intTrueNegative;
    private Integer intFalsePositive;
    private Integer intFalseNegative;
    private BigDecimal truePositive;
    private BigDecimal trueNegative;
    private BigDecimal falsePositive;
    private BigDecimal falseNegative;
    private BigDecimal nElements;
    private BigDecimal accuracy;
    private BigDecimal precision;
    private BigDecimal recall;
    private BigDecimal f1Score;
    public AssessmentMetrics() {
        this.intTruePositive = 0;
        this.intTrueNegative = 0;
        this.intFalsePositive = 0;
        this.intFalseNegative = 0;
        this.truePositive = BigDecimal.valueOf(0);
        this.trueNegative = BigDecimal.valueOf(0);
        this.falsePositive = BigDecimal.valueOf(0);
        this.falseNegative = BigDecimal.valueOf(0);
        this.nElements = BigDecimal.valueOf(0);
        this.accuracy = BigDecimal.valueOf(0);
        this.precision = BigDecimal.valueOf(0);
        this.recall = BigDecimal.valueOf(0);
        this.f1Score = BigDecimal.valueOf(0);
    }

    public BigDecimal getTruePositive() {
        return truePositive;
    }

    public void setTruePositive(BigDecimal truePositive) {
        this.truePositive = truePositive;
    }

    public BigDecimal getTrueNegative() {
        return trueNegative;
    }

    public void setTrueNegative(BigDecimal trueNegative) {
        this.trueNegative = trueNegative;
    }

    public BigDecimal getFalsePositive() {
        return falsePositive;
    }

    public void setFalsePositive(BigDecimal falsePositive) {
        this.falsePositive = falsePositive;
    }

    public BigDecimal getFalseNegative() {
        return falseNegative;
    }

    public void setFalseNegative(BigDecimal falseNegative) {
        this.falseNegative = falseNegative;
    }

    public BigDecimal getnElements() {
        return nElements;
    }

    public void setnElements(BigDecimal nElements) {
        this.nElements = nElements;
    }

    public BigDecimal getAccuracy() {
        return accuracy;
    }

    public void setAccuracy(BigDecimal accuracy) {
        this.accuracy = accuracy;
    }

    public BigDecimal getPrecision() {
        return precision;
    }

    public void setPrecision(BigDecimal precision) {
        this.precision = precision;
    }

    public BigDecimal getRecall() {
        return recall;
    }

    public void setRecall(BigDecimal recall) {
        this.recall = recall;
    }

    public BigDecimal getF1Score() {
        return f1Score;
    }

    public void setF1Score(BigDecimal f1Score) {
        this.f1Score = f1Score;
    }

    public Integer getIntTruePositive() {
        return intTruePositive;
    }

    public void setIntTruePositive(Integer intTruePositive) {
        this.intTruePositive = intTruePositive;
    }

    public Integer getIntTrueNegative() {
        return intTrueNegative;
    }

    public void setIntTrueNegative(Integer intTrueNegative) {
        this.intTrueNegative = intTrueNegative;
    }

    public Integer getIntFalsePositive() {
        return intFalsePositive;
    }

    public void setIntFalsePositive(Integer intFalsePositive) {
        this.intFalsePositive = intFalsePositive;
    }

    public Integer getIntFalseNegative() {
        return intFalseNegative;
    }

    public void setIntFalseNegative(Integer intFalseNegative) {
        this.intFalseNegative = intFalseNegative;
    }

    public Boolean cannotPrecision(){
        return truePositive.add(falsePositive).equals(BigDecimal.valueOf(0));
    }

    public Boolean cannotRecall(){
        return truePositive.add(falseNegative).equals(BigDecimal.valueOf(0));
    }

    public void increment(String element){
        switch (element){
            case "tp":
                this.intTruePositive++;
                break;
            case "tn":
                this.intTrueNegative++;
                break;
            case "fp":
                this.intFalsePositive++;
                break;
            case "fn":
                this.intFalseNegative++;
                break;
        }
    }

    public void generateMatrix(){
        Formatter fmt = new Formatter();
        fmt.format("%15s\n", "======= MATRIZ DE CONFUSÃO =======");
        fmt.format("%12s %12s\n", "pos", "neg");
        fmt.format("%12s %12s\n", getIntTruePositive(), getIntFalseNegative());
        fmt.format("%12s %12s\n", getIntFalsePositive(), getIntTrueNegative());

        System.out.println(fmt);
    }

    public void generateEvaluationMetrics(){
        this.truePositive = BigDecimal.valueOf(intTruePositive);
        this.trueNegative = BigDecimal.valueOf(intTrueNegative);
        this.falsePositive = BigDecimal.valueOf(intFalsePositive);
        this.falseNegative = BigDecimal.valueOf(intFalseNegative);

        System.out.println("======= MÉTRICAS DE AVALIAÇÃO =======");

        if(nElements.equals(BigDecimal.valueOf(0))){
            System.out.println("Não foi possível calcular a métrica Accuracy!");
        } else {
            setAccuracy(truePositive.add(trueNegative).divide(nElements, 4, BigDecimal.ROUND_HALF_UP));
            System.out.println("Accuracy: " + accuracy);
        }

        if(cannotPrecision()){
            System.out.println("Não foi possível calcular a métrica Precision!");
        } else {
            setPrecision(truePositive.divide(truePositive
                    .add(falsePositive), 4, BigDecimal.ROUND_HALF_UP));
            System.out.println("Precision: " + precision);
        }

        if(cannotRecall()){
            System.out.println("Não foi possível calcular a métrica Recall!");
        } else {
            setRecall(truePositive.divide(truePositive
                    .add(falseNegative), 4, BigDecimal.ROUND_HALF_UP));
            System.out.println("Recall: " + recall);
        }

        if(recall.equals(BigDecimal.valueOf(0.0000)) || recall.equals(BigDecimal.valueOf(0.0000))){
            System.out.println("Não foi possível calcular a métrica F1-Score!");
        } else {
            setF1Score(new BigDecimal(2).multiply(precision.multiply(recall))
                    .divide(precision.add(recall), 4, BigDecimal.ROUND_HALF_UP));
            System.out.println("F1-Score: " + f1Score);
        }
    }

}
