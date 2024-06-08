package com.edmilton.pln.model;

import com.edmilton.pln.enums.Label;
import com.edmilton.pln.enums.Sublabel;
import org.apache.commons.lang3.Range;

import java.util.HashMap;
import java.util.Map;

public class SublabelClassifier {
    private Label label;
    private int sub1 = 0;
    private int sub2 = 0;
    private int sub3 = 0;
    private int sub4 = 0;

    //Hashmap da quantidade de registros classificados corretamente por subníveis
    private Map<Integer, Sublabel> mapClassifier1 = new HashMap<>();
    private Map<Integer, Sublabel> mapClassifier2 = new HashMap<>();
    private Map<Integer, Sublabel> mapClassifier3 = new HashMap<>();
    private Map<Integer, Sublabel> mapClassifier4 = new HashMap<>();

    public SublabelClassifier(Label label) {
        this.label = label;
    }

    public Label getLabel() {
        return label;
    }

    public void setLabel(Label label) {
        this.label = label;
    }

    public Map<Integer, Sublabel> getMapClassifier1() {
        return mapClassifier1;
    }

    public Map<Integer, Sublabel> getMapClassifier2() {
        return mapClassifier2;
    }

    public Map<Integer, Sublabel> getMapClassifier3() {
        return mapClassifier3;
    }

    public Map<Integer, Sublabel> getMapClassifier4() {
        return mapClassifier4;
    }

    public void classify(Double similarity){
        if(similarity >= 0.0 || similarity <= 0.25) sub1++;
            else if(similarity > 0.25 || similarity <= 0.50) sub2++;
                else if(similarity > 0.50 || similarity <= 0.75) sub3++;
                    else if(similarity > 0.75) sub4++;
        if(!mapClassifier1.isEmpty()) mapClassifier1.clear();
        if(!mapClassifier2.isEmpty()) mapClassifier2.clear();
        if(!mapClassifier3.isEmpty()) mapClassifier3.clear();
        if(!mapClassifier4.isEmpty()) mapClassifier4.clear();

        mapClassifier1.put(sub1, Sublabel.LOW);
        mapClassifier2.put(sub2, Sublabel.NORMAL);
        mapClassifier3.put(sub3, Sublabel.MODERATE);
        mapClassifier4.put(sub4, Sublabel.HIGH);
    }

    private void exibir(Map<Integer, Sublabel> map){
        for(Map.Entry<Integer, Sublabel> entry : map.entrySet()){
            System.out.println(entry.getValue() + ": " + entry.getKey());
        }
    }

    public void exibirClassificacao(){
        System.out.println("===== CLASSIFICADOS CORRETAMENTE | " + "LABEL: " + this.label + " =====");
        this.exibir(mapClassifier1);
        this.exibir(mapClassifier2);
        this.exibir(mapClassifier3);
        this.exibir(mapClassifier4);
        System.out.println();
    }
}
