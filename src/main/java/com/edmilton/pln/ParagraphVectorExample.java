package com.edmilton.pln;

import com.edmilton.pln.model.AssessmentMetrics;
import com.edmilton.pln.model.SublabelClassifier;
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
    private static Logger log = LoggerFactory.getLogger(ParagraphVectorExample.class);

    public static final String RAIZ = "D:/Edmilton/tcc/tcc_atual - classificacao_emocoes/paragraphvectors";
    public static final String PATH_RESOURCES  = RAIZ + "/nlp_analise_sentimentos/sourceCode/cookbookapp/src/main/resources";
    public static final String PATH_LABEL = RAIZ + "/label";
    public static final String DIR_TESTE = RAIZ + "/teste/files";
    public static final String DIR_TREINO_NEG = RAIZ + "/treino/NEG";
    public static final String DIR_TREINO_POS = RAIZ + "/treino/POS";
    public static final String PATH_MODEL = RAIZ + "/modelos";
    public static final String DIR_CLASSIFICADOS_CORRETAMENTE = RAIZ + "/cc";

    public static void main(String[] args) throws IOException {
        SimpleDateFormat sdf = new SimpleDateFormat("HH:mm:ss", Locale.getDefault());
        Date hora = Calendar.getInstance().getTime(); // Ou qualquer outra forma que tem
        String horario = sdf.format(hora);
        log.info("Aplicação iniciada às {}", horario);

        AtomicLong tempoInicial = new AtomicLong(0);
        tempoInicial.set(System.currentTimeMillis());

        //int numFolds = 3;
        double percentTreino = 0.8;
        //double percentTeste = 0.2;

        int qtdCorretos = 0;

        AssessmentMetrics metrics = new AssessmentMetrics();

        //List<String> stopWords = FileUtils.readLines(new File(PATH_RESOURCES + "/stopwords-en.txt"), "utf-8");

        LabelAwareIterator labelAwareIterator = new FileLabelAwareIterator.Builder()
                .addSourceFolder(new File(PATH_LABEL))
                .build();

        //Alterando arraylist para hashset para preservar integridade dos dados,
        //ou seja, evitar a duplicidade
        HashSet<LabelledDocument> documentsHashSet = new HashSet<>();
        while(labelAwareIterator.hasNextDocument()){
            documentsHashSet.add(labelAwareIterator.nextDocument());
        }

        File diretorioTeste = new File(DIR_TESTE);
        File diretorioTreinoNeg = new File(DIR_TREINO_NEG);
        File diretorioTreinoPos = new File(DIR_TREINO_POS);

        List<String> nomesArquivosTeste = new ArrayList<>();
        List<Integer> indicesArquivosTeste = new ArrayList<>();

        //Depois de copiar os arquivos e separá-los em novos diretórios,
        //é necessário criar outro iterator apenas do novo diretório de arquivo de treino

        //Limite tamanho conjunto de dados teste
        /*int dadosTesteRestante = documents.size();
        int limiteDadosTeste = 0;*/

        SublabelClassifier classifierPos = new SublabelClassifier(Label.POS);
        SublabelClassifier classifierNeg = new SublabelClassifier(Label.NEG);
        int qtdArquivosTreino = 0;
        int qtdArquivosTeste = 0;


        //for(int i = 0; i < numFolds; i++){
        log.info("Iniciando execução...");
        qtdArquivosTreino = (int) Math.round(documentsHashSet.size() * percentTreino);
        qtdArquivosTeste = documentsHashSet.size() - qtdArquivosTreino;

            /*if(i == numFolds - 1) limiteDadosTeste = dadosTesteRestante;
            else limiteDadosTeste = documents.size() / numFolds;

            dadosTesteRestante = dadosTesteRestante - limiteDadosTeste;*/

        File[] arquivosTeste = diretorioTeste.listFiles(); //possiveis arquivos na pasta /resources/teste
        File[] arquivosTreinoNeg = diretorioTreinoNeg.listFiles(); //possiveis arquivos na pasta /resources/treino/neg
        File[] arquivosTreinoPos = diretorioTreinoPos.listFiles(); //possiveis arquivos na pasta /resources/treino/pos

        if(arquivosTeste != null){
            for (File file : arquivosTeste) file.delete();
        }

        if(arquivosTreinoNeg != null) {
            for(File file : arquivosTreinoNeg) file.delete();
        }

        if(arquivosTreinoPos != null) {
            for(File file : arquivosTreinoPos) file.delete();
        }

        File arquivoTeste = null;
        File arquivoTreino = null;

        int contArquivosTeste = 0;
        int contArquivosTreino = 0;

        //Transformando hashset em arraylist
        List<LabelledDocument> documentsArrayList = new ArrayList<>(documentsHashSet);

        if(documentsArrayList.size() != documentsArrayList.stream().distinct().count()){
            log.info("ATENÇÃO!!!!! Elementos duplicados no arrayList");
        } else {
            log.info("UFAAA!!! Não há elementos duplicados no arrayList");
        }

        //Dividindo os arquivos em subconjuntos de treino e teste
        for(int j = 0; j < documentsArrayList.size(); j++){

            //Stemizando o texto (reduzir cada palavra a seu radical)
            //String content = stemmingText(documents.get(j).getContent());
            Random rand = new Random();
            int min = 0;
            int max = documentsArrayList.size() - 1;
            int indice = rand.nextInt((max - min) + 1) + min;
            String conteudoTeste = null;
            String conteudoTreino = documentsArrayList.get(j).getContent();

            //Verificar se a lista de indices de teste contém o indice gerado aleatoriamente
            //e realizar os demais procedimentos

            while(true){
                if(!indicesArquivosTeste.isEmpty() && indicesArquivosTeste.contains(indice)){
                    indice = rand.nextInt((max - min) + 1) + min;
                } else {
                    break;
                }
            }

            conteudoTeste = documentsArrayList.get(indice).getContent();

            String nomeArquivoTeste = "teste"+j+".txt";
            String nomeArquivoTreino = "treino"+j+".txt";

            //Na primeira rodada, todos os arquivos são considerados para treino
            //A partir da segunda rodada é feita a divisão
            //i=0 significa a 1a rodada (da tentativa de implementar k-fold cross validation)
            //que portanto sempre criava arquivo de treino (qtdArquivosTreino = totalArquivos na 1a rodada)
            //e criava arquivos de teste apenas num total de totalArquivos / k
            //if(i == 0){
                    /*if(diretorioTeste.listFiles().length < limiteDadosTeste){
                        arquivoTeste = new File(diretorioTeste, nomeArquivoTeste);
                        criaArquivo(content, arquivoTeste);
                        contArquivosTeste++;
                    }
                    if(documents.get(j).getLabels().get(0).equals(Label.NEG.toString())) arquivoTreino = new File(diretorioTreinoNeg, nomeArquivoTreino);
                    else arquivoTreino = new File(diretorioTreinoPos, nomeArquivoTreino);
                    criaArquivo(content, arquivoTreino);
                    contArquivosTreino++;
                //} else {
                    if(!nomesArquivosTeste.contains(nomeArquivoTeste) && diretorioTeste.listFiles().length < limiteDadosTeste){
                        arquivoTeste = new File(diretorioTeste, nomeArquivoTeste); //arquivo de teste atual
                        nomesArquivosTeste.add(arquivoTeste.getName());
                        criaArquivo(content, arquivoTeste);
                        contArquivosTeste++;
                    } else {
                        if (documents.get(j).getLabels().get(0).equals("neg")) arquivoTreino = new File(diretorioTreinoNeg, nomeArquivoTreino);
                        else arquivoTreino = new File(diretorioTreinoPos, nomeArquivoTreino);
                        criaArquivo(content, arquivoTreino);
                        contArquivosTreino++;
                    }*/
            //}

            if(!nomesArquivosTeste.contains(nomeArquivoTeste) && Objects.requireNonNull(diretorioTeste.listFiles()).length < qtdArquivosTeste){
                arquivoTeste = new File(diretorioTeste, nomeArquivoTeste); //arquivo de teste atual
                nomesArquivosTeste.add(arquivoTeste.getName());
                criaArquivo(conteudoTeste, arquivoTeste);
                contArquivosTeste++;
            } else {
                if (documentsArrayList.get(j).getLabels().get(0).equals(Label.NEG.toString())) arquivoTreino = new File(diretorioTreinoNeg, nomeArquivoTreino);
                else arquivoTreino = new File(diretorioTreinoPos, nomeArquivoTreino);
                criaArquivo(conteudoTreino, arquivoTreino);
                contArquivosTreino++;
            }
        }

        if(contArquivosTeste == qtdArquivosTeste){
            log.info("Contador de arquivos teste é IGUAL a quantidade arquivos teste definida pelo percentual");
        } else {
            log.info("Contador de arquivos teste é DIFERENTE a quantidade arquivos teste definida pelo percentual");
        }

        log.info("Foram criados {} arquivos de teste e {} arquivos de treino.", contArquivosTeste, contArquivosTreino);

        Map<String, String> mapReal = new HashMap<>();
        Map<String, String> mapTest = new HashMap<>();
        ArrayList<String> idsUnlabelledDocument = new ArrayList<>();
        List<String> idsLabelledDocument = new ArrayList<>();
        Map<String, Double> mapSimilarity = new HashMap<>();

        FileLabelAwareIterator testeIterator = new FileLabelAwareIterator.Builder()
                .addSourceFolder(new File(RAIZ + "/teste"))
                .build();

        FileLabelAwareIterator treinoIterator = new FileLabelAwareIterator.Builder()
                .addSourceFolder(new File(RAIZ + "/treino"))
                .build();

        for (LabelledDocument real : documentsArrayList){
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
        int numEpochs = 2;
        int batch = 1000;
        int minWord = 1;

        /*if(filesModel.length > 0){
            log.info("Modelo existente! Iniciando carregamento...");
            AtomicLong timeSpent = new AtomicLong(0);
            timeSpent.set(System.currentTimeMillis());
            modelParagraphVectors = WordVectorSerializer.readParagraphVectors(new File(PATH_MODEL + "/modelo_" + System.currentTimeMillis() + ".zip"));
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
        } else {*/
            log.info("Iniciando configuração...");
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
        //}

        paragraphVectors.fit();

            /*for(File model : filesModel){
                model.delete(); //Excluindo o modelo salvo para salvar outro, como se fosse sobrescrever
            }*/
        WordVectorSerializer.writeParagraphVectors(paragraphVectors, new File(PATH_MODEL + "/modelo_" + System.currentTimeMillis() + ".zip"));
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
                result.add(new Pair<>(label.toUpperCase(), sim));
            }

            //Aqui devo pegar a chave do mapTest de unlabelledDocument.hashCode()
            //E criar uma iteração na lista 'result'
            //para verificar qual label de maior score do documento e atribuir ao mapTest
            Double s1 = Math.abs(result.get(0).getSecond());
            Double s2 = Math.abs(result.get(1).getSecond());
            if(mapTest.containsKey(unlabelledDocument.getContent().trim().toLowerCase())){
                mapReal.remove(unlabelledDocument.getContent().trim().toLowerCase());
            }
            if(s1 > s2){
                mapTest.put(unlabelledDocument.getContent().trim().toLowerCase(), result.get(0).getFirst());
                //pegar a pontuação para verificar a qual subclasse pertence
                mapSimilarity.put(unlabelledDocument.getContent().trim().toLowerCase(), s1);
            } else {
                mapTest.put(unlabelledDocument.getContent().trim().toLowerCase(), result.get(1).getFirst());
                //pegar a pontuação para verificar a qual subclasse pertence
                mapSimilarity.put(unlabelledDocument.getContent().trim().toLowerCase(), s2);
            }

        }

        for (int k = 0; k < mapTest.size(); k++){
            //Aqui preciso ver se há os documentos do mapTest
            //no mapReal, e se forem correspondentes, comparar o label
            if(mapReal.containsKey(idsUnlabelledDocument.get(k))){
                String labelReal = mapReal.get(idsUnlabelledDocument.get(k));
                String labelTest = mapTest.get(idsUnlabelledDocument.get(k));
                Double similarity = mapSimilarity.get(idsUnlabelledDocument.get(k));

                if(labelTest.equals(Label.POS.toString()) && labelReal.equals(Label.POS.toString())) {
                    classifierPos.classify(similarity);
                    metrics.increment("tp");
                }
                else if(labelTest.equals(Label.NEG.toString()) && labelReal.equals(Label.NEG.toString())) {
                    classifierNeg.classify(similarity);
                    metrics.increment("tn");
                }
                qtdCorretos++;
                String nameFile = "arqCorreto_0" + qtdCorretos + "_" + labelTest + "_" + similarity.toString().replace('.', '-') + ".txt"; //Ex.: correctly03_neg_0-65.txt
                File arquivoCorreto = new File(new File(DIR_CLASSIFICADOS_CORRETAMENTE), nameFile);
                criaArquivo(idsUnlabelledDocument.get(k), arquivoCorreto);
                if(labelTest.equals(Label.POS.toString()) && labelReal.equals(Label.NEG.toString())) metrics.increment("fp");
                else if(labelTest.equals(Label.NEG.toString()) && labelReal.equals(Label.POS.toString())) metrics.increment("fn");
            }
        }

        metrics.setnElements(BigDecimal.valueOf(qtdArquivosTeste));
        metrics.generateMatrix();

        metrics.generateEvaluationMetrics();

        classifierPos.exibirClassificacao();
        classifierNeg.exibirClassificacao();

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
