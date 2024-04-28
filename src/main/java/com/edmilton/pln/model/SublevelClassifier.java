package com.edmilton.pln.model;

import com.edmilton.pln.enums.Level;
import com.edmilton.pln.enums.Sublevel;
import org.apache.commons.lang3.Range;

import java.util.HashMap;
import java.util.Map;

public class SublevelClassifier {
    private Level level;
    private int sub1 = 0;
    private int sub2 = 0;
    private int sub3 = 0;
    private int sub4 = 0;

    //Hashmap da quantidade de registros classificados corretamente por subníveis
    private Map<Integer, Sublevel> mapClassifier1 = new HashMap<>();
    private Map<Integer, Sublevel> mapClassifier2 = new HashMap<>();
    private Map<Integer, Sublevel> mapClassifier3 = new HashMap<>();
    private Map<Integer, Sublevel> mapClassifier4 = new HashMap<>();

    public SublevelClassifier (Level level) {
        this.level = level;
    }

    public Level getLevel() {
        return level;
    }

    public void setLevel(Level level) {
        this.level = level;
    }

    public Map<Integer, Sublevel> getMapClassifier1() {
        return mapClassifier1;
    }

    public Map<Integer, Sublevel> getMapClassifier2() {
        return mapClassifier2;
    }

    public Map<Integer, Sublevel> getMapClassifier3() {
        return mapClassifier3;
    }

    public Map<Integer, Sublevel> getMapClassifier4() {
        return mapClassifier4;
    }

    public void classify(Double similarity){
        if(Range.between(0.0, 0.25).contains(similarity)) sub1++;
            else if(Range.between(0.25, 0.50).contains(similarity)) sub2++;
                else if(Range.between(0.50, 0.75).contains(similarity)) sub3++;
                    else if(Range.between(0.75, 1.0).contains(similarity)) sub4++;
        if(!mapClassifier1.isEmpty()) {
            mapClassifier1.clear();
            mapClassifier1.put(sub1, Sublevel.LOW);
        }
        if(!mapClassifier2.isEmpty()) {
            mapClassifier1.clear();
            mapClassifier1.put(sub2, Sublevel.NORMAL);
        }
        if(!mapClassifier3.isEmpty()) {
            mapClassifier1.clear();
            mapClassifier1.put(sub3, Sublevel.MODERATE);
        }
        if(!mapClassifier4.isEmpty()) {
            mapClassifier1.clear();
            mapClassifier1.put(sub4, Sublevel.HIGH);
        }
    }

    private void exibir(Map<Integer, Sublevel> map){
        for(Map.Entry<Integer, Sublevel> entry : map.entrySet()){
            System.out.println(entry.getKey() + ":" + entry.getValue() + "| Level " + this.level);
        }
    }

    public void exibirClassificacao(){
        this.exibir(mapClassifier1);
        this.exibir(mapClassifier2);
        this.exibir(mapClassifier3);
        this.exibir(mapClassifier4);
    }
}
