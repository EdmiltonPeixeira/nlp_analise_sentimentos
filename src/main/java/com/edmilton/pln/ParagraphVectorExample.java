package com.edmilton.pln;

import com.edmilton.pln.model.AssessmentMetrics;
import com.edmilton.pln.model.SublevelClassifier;
import org.apache.commons.lang3.Range;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.documentiterator.FileLabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tartarus.snowball.ext.EnglishStemmer;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.math.BigDecimal;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import com.edmilton.pln.enums.*;

public class ParagraphVectorExample {

    //Ele cria um criador de logs para esta classe, que pode ser usado para gerar mensagens de log.
    private static Logger log = LoggerFactory.getLogger(ParagraphVectorExample.class);

    public static final String PATH_RESOURCES  = "D:/Edmilton/tcc/tcc_atual - classificacao_emocoes/paragraphvectors/nlp_analise_sentimentos/src/main/resources";
    public static final String RESOURCE_TESTE = PATH_RESOURCES + "/teste/file";
    public static final String RESOURCE_TREINO_NEG = PATH_RESOURCES + "/treino/neg";
    public static final String RESOURCE_TREINO_POS = PATH_RESOURCES + "/treino/pos";
    public static final String PATH_MODEL = PATH_RESOURCES + "/model";
    public static final String PATH_CLASSIFIED_CORRECTLY = PATH_RESOURCES + "/classified_correctly";

    public static void main(String[] args) throws IOException {
        SimpleDateFormat sdf = new SimpleDateFormat("HH:mm:ss", Locale.getDefault());
        Date hora = Calendar.getInstance().getTime(); // Ou qualquer outra forma que tem
        String horario = sdf.format(hora);
        log.info("Aplicação iniciada às {}", horario);

        AtomicLong tempoInicial = new AtomicLong(0);
        tempoInicial.set(System.currentTimeMillis());

        int numFolds = 3;

        int countFilesCorrectly = 0;

        AssessmentMetrics metrics = new AssessmentMetrics();

        //List<String> stopWords = FileUtils.readLines(new File(PATH_RESOURCES + "/stopwords-en.txt"), "utf-8");

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

        SublevelClassifier classifierPos = new SublevelClassifier(Level.POS);
        SublevelClassifier classifierNeg = new SublevelClassifier(Level.NEG);

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

            //Dividindo os arquivos conforme k-fold cross validation
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
                    if(documents.get(j).getLabels().get(0).equals(Level.NEG.toString())) arquivoTreino = new File(diretorioTreinoNeg, nomeArquivoTreino);
                    else arquivoTreino = new File(diretorioTreinoPos, nomeArquivoTreino);
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
            Map<String, Double> mapSimilarity = new HashMap<String, Double>();

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


            ParagraphVectors modelParagraphVectors = null;
            ParagraphVectors paragraphVectors;

            double learning = 0.20;
            double minLearning = 0.1;
            int numEpochs = 200;
            int batch = 1000;
            int minWord = 1;

            if(filesModel.length > 0){
                log.info("Modelo existente! Iniciando carregamento...");
                AtomicLong timeSpent = new AtomicLong(0);
                timeSpent.set(System.currentTimeMillis());
                modelParagraphVectors = WordVectorSerializer.readParagraphVectors(new File(PATH_MODEL + "/model.zip"));
                paragraphVectors = new ParagraphVectors.Builder()
                        .learningRate(learning)
                        .minLearningRate(minLearning)
                        .batchSize(batch)
                        .epochs(numEpochs)
                        .minWordFrequency(minWord)
                        .iterate(treinoIterator)
                        .trainWordVectors(true)
                        .tokenizerFactory(tokenizerFactory)
                        .useExistingWordVectors(modelParagraphVectors)
                        .build();
                log.info("Carregamento e configuração concluído em {} ms", System.currentTimeMillis() - timeSpent.get());
            } else {
                log.info("Sem modelo existente para carregamento. Iniciando configuração...");
                paragraphVectors = new ParagraphVectors.Builder()
                        .learningRate(learning)
                        .minLearningRate(minLearning)
                        .batchSize(batch)
                        .epochs(numEpochs)
                        .minWordFrequency(minWord)
                        .iterate(treinoIterator)
                        .trainWordVectors(true)
                        .tokenizerFactory(tokenizerFactory)
                        .build();
                log.info("Configuração concluída!");
            }

            paragraphVectors.fit();

            for(File model : filesModel){
                model.delete(); //Excluindo o modelo salvo para salvar outro, como se fosse sobrescrever
            }
            WordVectorSerializer.writeParagraphVectors(paragraphVectors, new File(PATH_MODEL + "/model_" + numFolds + "folds.zip"));
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
                    //pegar a pontuação para verificar a qual subclasse pertence
                    mapSimilarity.put(unlabelledDocument.getContent().trim().toLowerCase(), result.get(0).getSecond());
                } else {
                    mapTest.put(unlabelledDocument.getContent().trim().toLowerCase(), result.get(1).getFirst());
                    //pegar a pontuação para verificar a qual subclasse pertence
                    mapSimilarity.put(unlabelledDocument.getContent().trim().toLowerCase(), result.get(1).getSecond());
                }

            }

            for (int k = 0; k < mapTest.size(); k++){
                //Aqui preciso ver se há os documentos do mapTest
                //no mapReal, e se forem correspondentes, comparar o label
                if(mapReal.containsKey(idsUnlabelledDocument.get(k))){
                    String labelReal = mapReal.get(idsUnlabelledDocument.get(k));
                    String labelTest = mapTest.get(idsUnlabelledDocument.get(k));
                    Double similarity = mapSimilarity.get(idsUnlabelledDocument.get(k));

                    if(labelTest.equals("pos") && labelReal.equals("pos")) {
                        classifierPos.classify(similarity);
                        metrics.increment("tp");
                    }
                    else if(labelTest.equals("neg") && labelReal.equals("neg")) {
                        classifierNeg.classify(similarity);
                        metrics.increment("tn");
                    }
                    countFilesCorrectly++;
                    String nameFile = "correctly0" + countFilesCorrectly + "_" + labelTest + "_" + similarity.toString().replace('.', '-') + ".txt"; //Ex.: correctly03_neg_0-65.txt
                    File fileCorrectly = new File(new File(PATH_CLASSIFIED_CORRECTLY), nameFile);
                    criaArquivo(idsUnlabelledDocument.get(k), fileCorrectly);
                    if(labelTest.equals("pos") && labelReal.equals("neg")) metrics.increment("fp");
                    else if(labelTest.equals("neg") && labelReal.equals("pos")) metrics.increment("fn");
                }
            }
        }

        metrics.setnElements(BigDecimal.valueOf(documents.size()));
        metrics.generateMatrix();
        System.out.println("Classificados corretamente: ");
        classifierPos.exibirClassificacao();
        classifierNeg.exibirClassificacao();

        metrics.generateEvaluationMetrics();

        SimpleDateFormat sdfFinal = new SimpleDateFormat("HH:mm:ss", Locale.getDefault());
        Date horaFinal = Calendar.getInstance().getTime(); // Ou qualquer outra forma que tem
        String horarioFinal = sdfFinal.format(horaFinal);
        log.info("Execução concluída às {}", horarioFinal);

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
        String[] words = text.split("\\s+");
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
