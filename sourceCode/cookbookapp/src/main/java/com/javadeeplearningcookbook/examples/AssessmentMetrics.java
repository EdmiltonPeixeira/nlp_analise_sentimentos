package com.javadeeplearningcookbook.examples;

import java.math.BigDecimal;

public class AssessmentMetrics {
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

    public Boolean cannotPrecision(){
        return truePositive.add(falsePositive).equals(BigDecimal.valueOf(0));
    }

    public Boolean cannotRecall(){
        return truePositive.add(falseNegative).equals(BigDecimal.valueOf(0));
    }
}
