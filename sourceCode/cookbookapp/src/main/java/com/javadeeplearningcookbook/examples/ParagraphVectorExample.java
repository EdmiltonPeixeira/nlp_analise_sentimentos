package com.javadeeplearningcookbook.examples;

import org.apache.arrow.flatbuf.Bool;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.sequencevectors.iterators.AbstractSequenceIterator;
import org.deeplearning4j.models.sequencevectors.transformers.impl.SentenceTransformer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.documentiterator.FileLabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tartarus.snowball.ext.EnglishStemmer;

import java.awt.*;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.math.BigDecimal;
import java.text.DecimalFormat;
import java.util.*;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

public class ParagraphVectorExample {

    //Ele cria um criador de logs para esta classe, que pode ser usado para gerar mensagens de log.
    private static Logger log = LoggerFactory.getLogger(ParagraphVectorExample.class);

    public static final String PATH_RESOURCES  = "D:/Edmilton/tcc/tcc_atual - classificacao_emocoes/paragraphvectors/nlp_analise_sentimentos/sourceCode/cookbookapp/src/main/resources";
    public static final String RESOURCE_TESTE = PATH_RESOURCES + "/teste/file";
    public static final String RESOURCE_TREINO_NEG = PATH_RESOURCES + "/treino/neg";
    public static final String RESOURCE_TREINO_POS = PATH_RESOURCES + "/treino/pos";
    public static final String PATH_MODEL = PATH_RESOURCES + "/model";

    public static void main(String[] args) throws IOException {
        AtomicLong tempoInicial = new AtomicLong(0);
        tempoInicial.set(System.currentTimeMillis());

        int numFolds = 4;

        int truePositives = 0; //previu corretamente como pos
        int falsePositives = 0; //previu incorretamente como pos
        int trueNegatives = 0; //previu corretamente como neg
        int falseNegatives = 0; //previu incorretamente como neg

        AssessmentMetrics metrics = new AssessmentMetrics();

        List<String> stopWords = FileUtils.readLines(new File(PATH_RESOURCES + "/stopwords-en.txt"), "utf-8");

        LabelAwareIterator labelAwareIterator = new FileLabelAwareIterator.Builder()
                .addSourceFolder(new File(PATH_RESOURCES + "/label"))
                .build();

        List<LabelledDocument> documents = new ArrayList<>();
        while(labelAwareIterator.hasNextDocument()){
            documents.add(labelAwareIterator.nextDocument());
        }

        File diretorioTeste = new File(RESOURCE_TESTE);
        File diretorioTreinoNeg = new File(RESOURCE_TREINO_NEG);
        File diretorioTreinoPos = new File(RESOURCE_TREINO_POS);

        List<String> listaArquivosTestados = new ArrayList<>();

        //Depois de copiar os arquivos e separá-los em novos diretórios,
        //é necessário criar outro iterator apenas do novo diretório de arquivo de treino

        //Limite tamanho conjunto de dados teste
        int dadosTesteRestante = documents.size();
        int limiteDadosTeste = 0;

        for(int i = 0; i < numFolds; i++){
            log.info("Iniciando rodada {} de {} da validação cruzada...", i+1, numFolds);

            if(i == numFolds - 1) limiteDadosTeste = dadosTesteRestante;
            else limiteDadosTeste = documents.size() / numFolds;

            dadosTesteRestante = dadosTesteRestante - limiteDadosTeste;

            File[] arquivosTeste = diretorioTeste.listFiles(); //possiveis arquivos na pasta /resources/teste
            File[] arquivosTreinoNeg = diretorioTreinoNeg.listFiles(); //possiveis arquivos na pasta /resources/treino/neg
            File[] arquivosTreinoPos = diretorioTreinoPos.listFiles(); //possiveis arquivos na pasta /resources/treino/pos
            if(arquivosTeste.length > 0) {
                for(File file : arquivosTeste) file.delete();
            }

            if(arquivosTreinoNeg.length > 0) {
                for(File file : arquivosTreinoNeg) file.delete();
            }

            if(arquivosTreinoPos.length > 0) {
                for(File file : arquivosTreinoPos) file.delete();
            }

            File arquivoTeste = null;
            File arquivoTreino = null;

            int countFileTest = 0;
            int countFileTrain = 0;

            for(int j = 0; j < documents.size(); j++){

                //Stemizando o texto (reduzir cada palavra a seu radical)
                //String content = stemmingText(documents.get(j).getContent());
                String content = documents.get(j).getContent();

                String nomeArquivoTeste = "teste"+j+".txt";
                String nomeArquivoTreino = "treino"+j+".txt";

                //Na primeira rodada, todos os arquivos são considerados para treino
                //A partir da segunda rodada é feita a divisão
                if(i == 0){
                    if(diretorioTeste.listFiles().length < limiteDadosTeste){
                        arquivoTeste = new File(diretorioTeste, nomeArquivoTeste);
                        criaArquivo(content, arquivoTeste);
                        countFileTest++;
                    }
                    if(documents.get(j).getLabels().get(0).equals("neg")){
                        arquivoTreino = new File(diretorioTreinoNeg, nomeArquivoTreino);
                    } else {
                        arquivoTreino = new File(diretorioTreinoPos, nomeArquivoTreino);
                    }
                    criaArquivo(content, arquivoTreino);
                    countFileTrain++;
                } else {
                    if(!listaArquivosTestados.contains(nomeArquivoTeste) && diretorioTeste.listFiles().length < limiteDadosTeste){
                        arquivoTeste = new File(diretorioTeste, nomeArquivoTeste); //arquivo de teste atual
                        listaArquivosTestados.add(arquivoTeste.getName());
                        criaArquivo(content, arquivoTeste);
                        countFileTest++;
                    } else {
                        if (documents.get(j).getLabels().get(0).equals("neg")) arquivoTreino = new File(diretorioTreinoNeg, nomeArquivoTreino);
                        else arquivoTreino = new File(diretorioTreinoPos, nomeArquivoTreino);
                        criaArquivo(content, arquivoTreino);
                        countFileTrain++;
                    }
                }
            }

            log.info("Foram criados {} arquivos de teste e {} arquivos de treino.", countFileTest, countFileTrain);

            Map<String, String> mapReal = new HashMap<String, String>();
            Map<String, String> mapTest = new HashMap<String, String>();
            List<String> idsUnlabelledDocument = new ArrayList<>();
            List<String> idsLabelledDocument = new ArrayList<>();

            FileLabelAwareIterator testeIterator = new FileLabelAwareIterator.Builder()
                    .addSourceFolder(new File(PATH_RESOURCES + "/teste"))
                    .build();

            FileLabelAwareIterator treinoIterator = new FileLabelAwareIterator.Builder()
                    .addSourceFolder(new File(PATH_RESOURCES + "/treino"))
                    .build();

            for (LabelledDocument real : documents){
                String id = real.getContent().trim().toLowerCase();
                String label = real.getLabels().get(0);

                mapReal.put(id, label);
                idsLabelledDocument.add(id);
            }

            //Um `TokenizerFactory` é criado para tokenizar o texto. O `CommonPreprocessor` é definido como o pré-processador de token,
            //que executa tarefas comuns de pré-processamento, como converter texto em letras minúsculas e remover caracteres especiais.
            TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
            tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

            File[] filesModel = new File(PATH_MODEL).listFiles();

            ParagraphVectors paragraphVectors;

            if(filesModel.length > 0){
                log.info("Modelo existente! Iniciando carregamento...");
                AtomicLong timeSpent = new AtomicLong(0);
                timeSpent.set(System.currentTimeMillis());
                paragraphVectors = WordVectorSerializer.readParagraphVectors(new File(PATH_MODEL + "/model.zip"));
                paragraphVectors.setTokenizerFactory(tokenizerFactory);
                SentenceTransformer transformer = new SentenceTransformer.Builder().iterator(treinoIterator)
                        .tokenizerFactory(tokenizerFactory).allowMultithreading(true)
                        .build();
                paragraphVectors.setSequenceIterator(new AbstractSequenceIterator.Builder<>(transformer).build());
                paragraphVectors.getConfiguration().setLearningRate(0.005);
                paragraphVectors.getConfiguration().setMinLearningRate(0.001);
                paragraphVectors.getConfiguration().setBatchSize(1000);
                paragraphVectors.getConfiguration().setEpochs(2);
                paragraphVectors.getConfiguration().setMinWordFrequency(1);
                //paragraphVectors.getConfiguration().setStopList(stopWords);
                paragraphVectors.getConfiguration().setTrainElementsVectors(true);
                log.info("Carregamento e ajuste de configuração concluído em {} ms", System.currentTimeMillis() - timeSpent.get());
            } else {
                //Nestas linhas, um modelo `ParagraphVectors` é construído. É configurado com vários parâmetros, como taxa de aprendizagem,
                //tamanho do lote e o número de épocas de treinamento. Ele é treinado usando `labelAwareIterator` e `tokenizerFactory` que você criou anteriormente.
                log.info("Sem modelo existente para carregamento. Iniciando configuração...");
                paragraphVectors = new ParagraphVectors.Builder()
                        .learningRate(0.005)
                        .minLearningRate(0.001)
                        .batchSize(1000)
                        .epochs(2)
                        .minWordFrequency(1)
                        .iterate(treinoIterator)
                        //.stopWords(stopWords)
                        .trainWordVectors(true)
                        .tokenizerFactory(tokenizerFactory)
                        .build();
                log.info("Configuração concluída!");
            }

            paragraphVectors.fit();

            for(File model : filesModel){
                model.delete(); //Excluindo o modelo salvo para salvar outro, como se fosse sobrescrever
            }
            WordVectorSerializer.writeParagraphVectors(paragraphVectors, new File(PATH_MODEL + "/model.zip"));
            log.info("Modelo salvo com sucesso!");

            //O `InMemoryLookupTable` é recuperado do modelo `paragraphVectors`. Esta tabela contém incorporações de palavras aprendidas durante o treinamento.
            InMemoryLookupTable<VocabWord> lookupTable = (InMemoryLookupTable<VocabWord>)paragraphVectors.getLookupTable();

            //Este loop percorre cada documento não rotulado no `unClassifiedIterator`

            while (testeIterator.hasNextDocument()) {

                //Para cada documento, seu conteúdo é tokenizado usando `tokenizerFactory`, e os tokens são armazenados em `documentAsTokens`.
                LabelledDocument unlabelledDocument = testeIterator.nextDocument();
                List<String> documentAsTokens = tokenizerFactory.create(unlabelledDocument.getContent()).getTokens();

                idsUnlabelledDocument.add(unlabelledDocument.getContent().trim().toLowerCase());

                //O código abaixo conta os tokens que existem no vocabulário de `lookupTable`.
                VocabCache vocabCache = lookupTable.getVocab();
                AtomicInteger cnt = new AtomicInteger(0);
                for (String word: documentAsTokens) {
                    if (vocabCache.containsWord(word)){
                        cnt.incrementAndGet();
                    }
                }

                //Este código cria um `INDArray` chamado `allWords` para armazenar os vetores das palavras no documento.
                INDArray allWords = Nd4j.create(cnt.get(), lookupTable.layerSize());
                cnt.set(0);
                for (String word: documentAsTokens) {
                    if (vocabCache.containsWord(word))
                        allWords.putRow(cnt.getAndIncrement(), lookupTable.vector(word));
                }

                //O `documentVector` é calculado como a média de todos os vetores de palavras no documento.
                INDArray documentVector = allWords.mean(0);

                List<String> labels = paragraphVectors.getLabelsSource().getLabels();

                //Esta parte calcula a similaridade de cosseno entre o vetor de documento e os vetores de rótulo para cada rótulo possível.
                List<Pair<String, Double>> result = new ArrayList<>();
                for (String label: labels) {
                    INDArray vecLabel = lookupTable.vector(label);
                    if (vecLabel == null){
                        throw new IllegalStateException("Label '"+ label+"' has no known vector!");
                    }
                    double sim = Transforms.cosineSim(documentVector, vecLabel);
                    result.add(new Pair<String, Double>(label, sim));
                }

                //Aqui devo pegar a chave do mapTest de unlabelledDocument.hashCode()
                //E criar uma iteração na lista 'result'
                //para verificar qual label de maior score do documento e atribuir ao mapTest
                if(result.get(0).getSecond() > result.get(1).getSecond()){
                    mapTest.put(unlabelledDocument.getContent().trim().toLowerCase(), result.get(0).getFirst());
                } else {
                    mapTest.put(unlabelledDocument.getContent().trim().toLowerCase(), result.get(1).getFirst());
                }

            }

            for (int k = 0; k < mapTest.size(); k++){
                //Aqui preciso ver se há os documentos do mapTest
                //no mapReal, e se forem correspondentes, comparar o label
                if(mapReal.containsKey(idsUnlabelledDocument.get(k))){
                    String labelReal = mapReal.get(idsUnlabelledDocument.get(k));
                    String labelTest = mapTest.get(idsUnlabelledDocument.get(k));

                    if(labelTest.equals("pos") && labelReal.equals("pos")) truePositives++;
                    else if(labelTest.equals("neg") && labelReal.equals("neg")) trueNegatives++;
                    else if(labelTest.equals("pos") && labelReal.equals("neg")) falsePositives++;
                    else if(labelTest.equals("neg") && labelReal.equals("pos")) falseNegatives++;
                }
            }
        }

        System.out.println("======= MATRIZ DE CONFUSÃO =======");
        System.out.println("     pos       neg");
        System.out.println("       "+truePositives+"          "+falseNegatives);
        System.out.println("       "+falsePositives+"          "+trueNegatives);

        metrics.setTruePositive(BigDecimal.valueOf(truePositives));
        metrics.setTrueNegative(BigDecimal.valueOf(trueNegatives));
        metrics.setFalsePositive(BigDecimal.valueOf(falsePositives));
        metrics.setFalseNegative(BigDecimal.valueOf(falseNegatives));
        metrics.setnElements(BigDecimal.valueOf(documents.size()));

        System.out.println("======= MÉTRICAS DE AVALIAÇÃO =======");

        if(metrics.getnElements().equals(BigDecimal.valueOf(0))){
            System.out.println("Não foi possível calcular a métrica Accuracy!");
        } else {
            metrics.setAccuracy(metrics.getTruePositive().add(metrics.getTrueNegative()).divide(metrics.getnElements(), 4, BigDecimal.ROUND_HALF_UP));
            System.out.println("Accuracy: " + metrics.getAccuracy());
        }

        if(metrics.cannotPrecision()){
            System.out.println("Não foi possível calcular a métrica Precision!");
        } else {
            metrics.setPrecision(metrics.getTruePositive().divide(metrics.getTruePositive()
                    .add(metrics.getFalsePositive()), 4, BigDecimal.ROUND_HALF_UP));
            System.out.println("Precision: " + metrics.getPrecision());
        }

        if(metrics.cannotRecall()){
            System.out.println("Não foi possível calcular a métrica Recall!");
        } else {
            metrics.setRecall(metrics.getTruePositive().divide(metrics.getTruePositive()
                    .add(metrics.getFalseNegative()), 4, BigDecimal.ROUND_HALF_UP));
            System.out.println("Recall: " + metrics.getRecall());
        }

        //Criar uma função em AssessmentMetrics para verificar se é possível
        //calcular o F1-Score, devido o erro ao dividir por zero.
        //As condições codificadas abaixo resolveram o problema, garantindo
        //que não tentasse realizar o cálculo. Não estava sendo verificado o valor 0.0, somente 0.
        if(metrics.getRecall().equals(BigDecimal.valueOf(0.0000)) || metrics.getRecall().equals(BigDecimal.valueOf(0.0000))){
            System.out.println("Não foi possível calcular a métrica F1-Score!");
        } else {
            metrics.setF1Score(new BigDecimal(2).multiply(metrics.getPrecision().multiply(metrics.getRecall()))
                    .divide(metrics.getPrecision().add(metrics.getRecall()), 4, BigDecimal.ROUND_HALF_UP));
            System.out.println("F1-Score: " + metrics.getF1Score());
        }

        //BigDecimal tpRate = bigTp.divide(bigTp.add(bigFn), 2, BigDecimal.ROUND_HALF_UP);
        //BigDecimal tnRate = bigTn.divide(bigTn.add(bigFp), 2, BigDecimal.ROUND_HALF_UP);
        //BigDecimal fpRate = bigFp.divide(bigFp.add(bigTn), 2, BigDecimal.ROUND_HALF_UP);
        //BigDecimal fnRate = bigFn.divide(bigFn.add(bigTp), 2, BigDecimal.ROUND_HALF_UP);
        //System.out.println("TP Rate: " + tpRate);
        //.out.println("TN Rate: " + tnRate);
        //System.out.println("FP Rate: " + fpRate);
        //System.out.println("FN Rate: " + fnRate);
        AtomicLong tempoTotal = new AtomicLong(0);
        tempoTotal.set(System.currentTimeMillis() - tempoInicial.get());

        long HH = TimeUnit.MILLISECONDS.toHours(tempoTotal.get());
        long MM = TimeUnit.MILLISECONDS.toMinutes(tempoTotal.get());
        long SS = TimeUnit.MILLISECONDS.toSeconds(tempoTotal.get());

        log.info("Tempo total de execução: {}:{}:{}", HH, MM, SS);
    }

    public static void criaArquivo(String content, File file){
        try{
            FileWriter writer = new FileWriter(file); //Criando um escritor para o arquivo txt
            //String content = labelledDocument.getContent(); //Criando variável para ajustar o conteúdo antes de passar ao arquivo
            //String quebraDeLinha = System.lineSeparator();
            //if(content.contains(".")) content.replace(".", "." + quebraDeLinha); //Adicionando quebra de linha a cada . no texto
            writer.write(content); //Escrevendo o texto no arquivo txt
            writer.close();
            file.createNewFile(); //Criando o arquivo no sistema
            //log.info("Arquivo " + file.getName() + " criado com sucesso.");
        } catch (IOException e){
            e.printStackTrace();
        }
    }

    public static String stemmingText(String text){
        String[] words = text.split("//s+");
        EnglishStemmer englishStemmer = new EnglishStemmer();
        StringBuilder stemmedText = new StringBuilder();

        for(String word : words){
            englishStemmer.setCurrent(word);
            if(englishStemmer.stem()){
                stemmedText.append(englishStemmer.getCurrent()).append(" ");
            } else {
                stemmedText.append(word).append(" ");
            }
        }

        String finalStemmedText = stemmedText.toString().trim();

        return finalStemmedText;
    }
}

//Essa é a explicação detalhada do código. Abrange a configuração, o treinamento e o uso de um modelo de vetor de parágrafo
//para classificação de texto. A parte principal está em como ele representa documentos e rótulos como vetores e mede sua
//similaridade usando similaridade de cosseno.
