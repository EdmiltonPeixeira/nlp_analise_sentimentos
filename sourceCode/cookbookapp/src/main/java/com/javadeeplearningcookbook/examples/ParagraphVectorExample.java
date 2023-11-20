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

import java.awt.*;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.math.BigDecimal;
import java.text.DecimalFormat;
import java.util.*;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

public class ParagraphVectorExample {

    //Ele cria um criador de logs para esta classe, que pode ser usado para gerar mensagens de log.
    private static Logger log = LoggerFactory.getLogger(ParagraphVectorExample.class);

    public static final String PATH_RESOURCES  = "D:/Edmilton/tcc/tcc_atual - classificacao_emocoes/paragraphvectors/nlp_analise_sentimentos/sourceCode/cookbookapp/src/main/resources";
    public static final String RESOURCE_TESTE = PATH_RESOURCES + "/teste/file";
    public static final String RESOURCE_TREINO_SAD = PATH_RESOURCES + "/treino/sad";
    public static final String RESOURCE_TREINO_HAPPY = PATH_RESOURCES + "/treino/happy";
    public static final String PATH_MODEL = PATH_RESOURCES + "/model";

    public static void main(String[] args) throws IOException {

        //declarar os elementos da matriz de confusão
        //no início da classe principal, para gerar assim apenas
        //uma matriz de confusão, que seria a soma da matriz de cada iteração k,
        //e também permitir o cálculo de acurácia, precisão...
        //porque atualmente está sendo gerada uma matriz a cada iteração k
        int truePositives = 0; //previu corretamente como happy
        int falsePositives = 0; //previu incorretamente como happy
        int trueNegatives = 0; //previu corretamente como sad
        int falseNegatives = 0; //previu incorretamente como sad

        AssessmentMetrics metrics = new AssessmentMetrics();

        List<String> stopWords = FileUtils.readLines(new File(PATH_RESOURCES + "/stopwords-en.txt"), "utf-8");

        //Iterator modificado
        //ClassPathResource classifiedResource = new ClassPathResource("label");
        //tentar passar a pasta label como file no .addSourceFolder abaixo
        LabelAwareIterator labelAwareIterator = new FileLabelAwareIterator.Builder()
                .addSourceFolder(new File(PATH_RESOURCES + "/label"))
                .build();

        //Sendo k = 4 ou k = 10, utilizar apenas um arquivo para teste
        //e os demais para treinar o modelo.
        //Verificar quantos arquivos existem no diretório "label". Depois, pegar 1 arquivo para teste
        //e k-1 arquivos para treino, até que todos tenham sido utilizados para treino e para teste.

        //Criar um laço para iterar seguindo a quantidade de arquivos em "label",
        //realizando todos os passos do paragraphvector por k vezes.
        //Exemplo: Sendo k = 4, na primeira iteração deve-se utilizar os documentos 2, 3 e 4 para treino
        //e o documento 1 para teste.

        //Quando passar o iterator para o paragraphvector, terá que ser apenas de k-1 documentos

        //Criando um array de LabelledDocument, que são os arquivos da pasta "label".

        //Depois, para aplicar o k-fold, devemos criar outro diretório com uma cópia dos arquivos de treino
        //dividindo-os em 1 para teste e k-1 para treino

        List<LabelledDocument> documents = new ArrayList<>();
        while(labelAwareIterator.hasNextDocument()){
            documents.add(labelAwareIterator.nextDocument());
        }

        File diretorioTeste = new File(RESOURCE_TESTE);
        File diretorioTreinoSad = new File(RESOURCE_TREINO_SAD);
        File diretorioTreinoHappy = new File(RESOURCE_TREINO_HAPPY);

        List<String> listaArquivosTestados = new ArrayList<>();

        //Depois de copiar os arquivos e separá-los em novos diretórios,
        //é necessário criar outro iterator apenas do novo diretório de arquivo de treino

        for(int i = 0; i < documents.size(); i++){
            File[] arquivosTeste = diretorioTeste.listFiles(); //possiveis arquivos na pasta /resources/teste
            File[] arquivosTreinoSad = diretorioTreinoSad.listFiles(); //possiveis arquivos na pasta /resources/treino/sad
            File[] arquivosTreinoHappy = diretorioTreinoHappy.listFiles(); //possiveis arquivos na pasta /resources/treino/happy
            if(arquivosTeste.length > 0) {
                for(File file : arquivosTeste) file.delete(); //deixando a pasta /teste vazia para uma nova iteração,
            }                                                 //porque deve ser testado apenas um arquivo por vez

            if(arquivosTreinoSad.length > 0) {
                for(File file : arquivosTreinoSad) file.delete();
            }

            if(arquivosTreinoHappy.length > 0) {
                for(File file : arquivosTreinoHappy) file.delete();
            }

            for(int j = 0; j < documents.size(); j++){
                //Antes de criar os novos arquivos de teste e treino
                //deve ser verificado se o diretório de teste está vazio,
                //se não, todos os arquivos de teste devem ser excluídos.

                String nomeArquivoTeste = "teste0"+j+".txt";
                String nomeArquivoTreino = "treino0"+j+".txt";

                if(!listaArquivosTestados.contains(nomeArquivoTeste) && diretorioTeste.listFiles().length == 0){
                    File arquivoTeste = new File(diretorioTeste, nomeArquivoTeste); //arquivo de teste atual
                    listaArquivosTestados.add(arquivoTeste.getName());
                    criaArquivo(documents.get(j), arquivoTeste);
                } else {
                    if (documents.get(j).getLabels().get(0).equals("sad")){
                        File arquivoTreino = new File(diretorioTreinoSad, nomeArquivoTreino);
                        criaArquivo(documents.get(j), arquivoTreino);
                    } else {
                        File arquivoTreino = new File(diretorioTreinoHappy, nomeArquivoTreino);
                        criaArquivo(documents.get(j), arquivoTreino);
                    }
                }
            }

            // 1 - Criar dois map<chave,valor> para armazenar <identificador, label> dos arquivos de treino e de teste
            Map<Integer, String> mapReal = new HashMap<Integer, String>();
            Map<Integer, String> mapTest = new HashMap<Integer, String>();
            List<Integer> idsUnlabelledDocument = new ArrayList<>();
            List<Integer> idsLabelledDocument = new ArrayList<>();
            // 2 - Ao final do treinamento, depois de atribuídos os scores,
            //verificar o label de maior score para cada documento e setar no map de treino mencionado no passo 1

            // 3 - Comparar o map de treino com o de teste e incrementar elementos da matriz de confusão

            //Aqui, um `LabelAwareIterator` é criado para ler documentos do diretório teste e treino, dentro do diretório resource
            //ClassPathResource testeResource = new ClassPathResource("teste");
            FileLabelAwareIterator testeIterator = new FileLabelAwareIterator.Builder()
                    .addSourceFolder(new File(PATH_RESOURCES + "/teste"))
                    .build();

            //ClassPathResource treinoResource = new ClassPathResource("treino");
            FileLabelAwareIterator treinoIterator = new FileLabelAwareIterator.Builder()
                    .addSourceFolder(new File(PATH_RESOURCES + "/treino"))
                    .build();

            //Preenchendo o mapReal
            for (LabelledDocument real : documents){
                Integer id = real.getContent().length();
                String label = real.getLabel();

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
                paragraphVectors.getConfiguration().setLearningRate(0.025);
                paragraphVectors.getConfiguration().setMinLearningRate(0.001);
                paragraphVectors.getConfiguration().setBatchSize(1000);
                paragraphVectors.getConfiguration().setEpochs(100);
                paragraphVectors.getConfiguration().setMinWordFrequency(2);
                paragraphVectors.getConfiguration().setStopList(stopWords);
                paragraphVectors.getConfiguration().setTrainElementsVectors(true);
                log.info("Carregamento e ajuste de configuração concluído em {} ms", System.currentTimeMillis() - timeSpent.get());
            } else {
                //Nestas linhas, um modelo `ParagraphVectors` é construído. É configurado com vários parâmetros, como taxa de aprendizagem,
                //tamanho do lote e o número de épocas de treinamento. Ele é treinado usando `labelAwareIterator` e `tokenizerFactory` que você criou anteriormente.
                log.info("Sem modelo existente para carregamento. Iniciando configuração...");
                paragraphVectors = new ParagraphVectors.Builder()
                        .learningRate(0.025)
                        .minLearningRate(0.001)
                        .batchSize(1000)
                        .epochs(100)
                        .minWordFrequency(2)
                        .iterate(treinoIterator)
                        .stopWords(stopWords)
                        .trainWordVectors(true)
                        .tokenizerFactory(tokenizerFactory)
                        .build();
            }

            paragraphVectors.fit();

            for(File model : filesModel){
                model.delete(); //Excluindo o modelo salvo para salvar outro, como se fosse sobrescrever
            }
            WordVectorSerializer.writeParagraphVectors(paragraphVectors, new File(PATH_MODEL + "/model.zip"));

            //O `InMemoryLookupTable` é recuperado do modelo `paragraphVectors`. Esta tabela contém incorporações de palavras aprendidas durante o treinamento.
            InMemoryLookupTable<VocabWord> lookupTable = (InMemoryLookupTable<VocabWord>)paragraphVectors.getLookupTable();

            //Este loop percorre cada documento não rotulado no `unClassifiedIterator`

            while (testeIterator.hasNextDocument()) {

                //Para cada documento, seu conteúdo é tokenizado usando `tokenizerFactory`, e os tokens são armazenados em `documentAsTokens`.
                LabelledDocument unlabelledDocument = testeIterator.nextDocument();
                List<String> documentAsTokens = tokenizerFactory.create(unlabelledDocument.getContent()).getTokens();

                idsUnlabelledDocument.add(unlabelledDocument.getContent().length()-1);

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

                List<String> labels = treinoIterator.getLabelsSource().getLabels();

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
                    mapTest.put(unlabelledDocument.getContent().length()-1, result.get(0).getFirst());
                } else {
                    mapTest.put(unlabelledDocument.getContent().length()-1, result.get(1).getFirst());
                }

            }

            for (int k = 0; k < mapTest.size(); k++){
                //Aqui preciso ver se há os documentos do mapTest
                //no mapReal, e se forem correspondentes, comparar o label
                if(mapReal.containsKey(idsUnlabelledDocument.get(k))){
                    String labelReal = mapReal.get(idsUnlabelledDocument.get(k));
                    String labelTest = mapTest.get(idsUnlabelledDocument.get(k));

                    if(labelTest.equals("happy") && labelReal.equals("happy")) truePositives++;
                    else if(labelTest.equals("sad") && labelReal.equals("sad")) trueNegatives++;
                    else if(labelTest.equals("happy") && labelReal.equals("sad")) falsePositives++;
                    else if(labelTest.equals("sad") && labelReal.equals("happy")) falseNegatives++;
                }
            }
        }

        System.out.println("======= MATRIZ DE CONFUSÃO =======");
        System.out.println("     happy       sad");
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
            metrics.setAccuracy(metrics.getTruePositive().add(metrics.getTrueNegative()).divide(metrics.getnElements(), 2, BigDecimal.ROUND_HALF_UP));
            System.out.println("Accuracy: " + metrics.getAccuracy());
        }

        if(metrics.cannotPrecision()){
            System.out.println("Não foi possível calcular a métrica Precision!");
        } else {
            metrics.setPrecision(metrics.getTruePositive().divide(metrics.getTruePositive()
                    .add(metrics.getFalsePositive()), 2, BigDecimal.ROUND_HALF_UP));
            System.out.println("Precision: " + metrics.getPrecision());
        }

        if(metrics.cannotRecall()){
            System.out.println("Não foi possível calcular a métrica Recall!");
        } else {
            metrics.setRecall(metrics.getTruePositive().divide(metrics.getTruePositive()
                    .add(metrics.getFalseNegative()), 2, BigDecimal.ROUND_HALF_UP));
            System.out.println("Recall: " + metrics.getRecall());
        }

        //Criar uma função em AssessmentMetrics para verificar se é possível
        //calcular o F1-Score, devido o erro ao dividir por zero.
        //As condições codificadas abaixo resolveram o problema, garantindo
        //que não tentasse realizar o cálculo. Não estava sendo verificado o valor 0.0, somente 0.
        if(metrics.getRecall().equals(BigDecimal.valueOf(0.00)) || metrics.getRecall().equals(BigDecimal.valueOf(0.00))){
            System.out.println("Não foi possível calcular a métrica F1-Score!");
        } else {
            metrics.setF1Score(new BigDecimal(2).multiply(metrics.getPrecision().multiply(metrics.getRecall()))
                    .divide(metrics.getPrecision().add(metrics.getRecall()), 2, BigDecimal.ROUND_HALF_UP));
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
    }

    public static void criaArquivo(LabelledDocument labelledDocument, File file){
        try{
            FileWriter writer = new FileWriter(file); //Criando um escritor para o arquivo txt
            String content = labelledDocument.getContent(); //Criando variável para ajustar o conteúdo antes de passar ao arquivo
            //String quebraDeLinha = System.lineSeparator();
            //if(content.contains(".")) content.replace(".", "." + quebraDeLinha); //Adicionando quebra de linha a cada . no texto
            writer.write(content); //Escrevendo o texto no arquivo txt
            writer.close();
            file.createNewFile(); //Criando o arquivo no sistema
            System.out.println("Arquivo " + file.getName() + " criado com sucesso.");
        } catch (IOException e){
            e.printStackTrace();
        }
    }
}

//Essa é a explicação detalhada do código. Abrange a configuração, o treinamento e o uso de um modelo de vetor de parágrafo
//para classificação de texto. A parte principal está em como ele representa documentos e rótulos como vetores e mede sua
//similaridade usando similaridade de cosseno.
