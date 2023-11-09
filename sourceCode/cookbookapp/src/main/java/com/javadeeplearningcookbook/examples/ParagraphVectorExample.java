package com.javadeeplearningcookbook.examples;

import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
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
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.concurrent.atomic.AtomicInteger;

public class ParagraphVectorExample {

    //Ele cria um criador de logs para esta classe, que pode ser usado para gerar mensagens de log.
    private static Logger log = LoggerFactory.getLogger(ParagraphVectorExample.class);

    public static final String PATH_RESOURCES  = "D:/Edmilton/tcc/tcc_atual - classificacao_emocoes/paragraphvectors/05_Implementing_NLP - Testando alterações/sourceCode/cookbookapp/src/main/resources";
    public static final String RESOURCE_TESTE = PATH_RESOURCES + "/teste";
    public static final String RESOURCE_TREINO = PATH_RESOURCES + "/treino";

    public static void main(String[] args) throws IOException {

        //Iterator modificado
        ClassPathResource classifiedResource = new ClassPathResource("label");
        LabelAwareIterator labelAwareIterator = new FileLabelAwareIterator.Builder()
                .addSourceFolder(classifiedResource.getFile())
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

        //Depois de copiar os arquivos e separá-los em novos diretórios,
        //é necessário criar outro iterator apenas do novo diretório de arquivo de treino

        for(int i = 0; i < documents.size(); i++){
            /* happy - 1 - positive
               sad - 0 - negative

               k = 4 -> [n0, n1, n2, n3]
               iteração 1:
               teste  -> n0
               treino -> n1, n2, n3
            */

            File diretorioTeste = new File(RESOURCE_TESTE);
            File diretorioTreino = new File(RESOURCE_TREINO);

            List<String> listaArquivosTestados = new ArrayList<>();

            File[] arquivosTeste = diretorioTeste.listFiles(); //possiveis arquivos na pasta /resources/teste
            File[] arquivosTreino = diretorioTreino.listFiles(); //possiveis arquivos na pasta /resources/treino
            if(arquivosTeste.length > 0) {
                for(File file : arquivosTeste) file.delete(); //deixando a pasta /teste vazia para uma nova iteração,
            }                                                 //porque deve ser testado apenas um arquivo por vez

            if(arquivosTreino.length > 0) {
                for(File file : arquivosTreino) file.delete();
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
                    File arquivoTreino = new File(diretorioTreino, nomeArquivoTreino);
                    criaArquivo(documents.get(j), arquivoTreino);
                }

            }

            int truePositives = 0; //previu corretamente como happy
            int falsePositives = 0; //previu incorretamente como happy
            int trueNegatives = 0; //previu corretamente como sad
            int falseNegatives = 0; //previu incorretamente como sad

            // 1 - Criar dois map<chave,valor> para armazenar <identificador, label> dos arquivos de treino e de teste
            Map<Integer, String> mapTrain = new HashMap<Integer, String>();
            Map<Integer, String> mapTest = new HashMap<Integer, String>();
            List<Integer> idsUnlabelledDocument = new ArrayList<>();
            List<Integer> idsLabelledDocument = new ArrayList<>();
            // 2 - Ao final do treinamento, depois de atribuídos os scores,
            //verificar o label de maior score para cada documento e setar no map de treino mencionado no passo 1

            // 3 - Comparar o map de treino com o de teste e incrementar elementos da matriz de confusão

            //Aqui, um `LabelAwareIterator` é criado para ler documentos do diretório teste e treino, dentro do diretório resource
            ClassPathResource testeResource = new ClassPathResource("teste");
            LabelAwareIterator testeIterator = new FileLabelAwareIterator.Builder()
                    .addSourceFolder(classifiedResource.getFile())
                    .build();

            ClassPathResource treinoResource = new ClassPathResource("treino");
            LabelAwareIterator treinoIterator = new FileLabelAwareIterator.Builder()
                    .addSourceFolder(classifiedResource.getFile())
                    .build();

            //Preenchendo o mapTrain
            while (treinoIterator.hasNextDocument()){
                LabelledDocument labelledDocument = treinoIterator.nextDocument();
                Integer id = labelledDocument.getContent().length();
                String label = labelledDocument.getLabel();

                mapTrain.put(id, label);
                idsLabelledDocument.add(id);
            }

            //Um `TokenizerFactory` é criado para tokenizar o texto. O `CommonPreprocessor` é definido como o pré-processador de token,
            //que executa tarefas comuns de pré-processamento, como converter texto em letras minúsculas e remover caracteres especiais.
            TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
            tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());


            //Nestas linhas, um modelo `ParagraphVectors` é construído. É configurado com vários parâmetros, como taxa de aprendizagem,
            //tamanho do lote e o número de épocas de treinamento. Ele é treinado usando `labelAwareIterator` e `tokenizerFactory` que você criou anteriormente.
            ParagraphVectors paragraphVectors = new ParagraphVectors.Builder()
                    .learningRate(0.050)
                    .minLearningRate(0.005)
                    .batchSize(1000)
                    .epochs(2000)
                    .iterate(treinoIterator)
                    .trainWordVectors(true)
                    .tokenizerFactory(tokenizerFactory)
                    .build();
            paragraphVectors.fit();

            //Aqui, um novo `LabelAwareIterator` é criado para documentos não rotulados de uma pasta de origem especificada como "não rotulada".
            //ATUALIZANDO: o iterator de arquivo de teste já foi criado acima

            /*ClassPathResource unClassifiedResource = new ClassPathResource("unlabeled");
            FileLabelAwareIterator unClassifiedIterator = new FileLabelAwareIterator.Builder()
                    .addSourceFolder(unClassifiedResource.getFile())
                    .build();*/

            //O `InMemoryLookupTable` é recuperado do modelo `paragraphVectors`. Esta tabela contém incorporações de palavras aprendidas durante o treinamento.
            InMemoryLookupTable<VocabWord> lookupTable = (InMemoryLookupTable<VocabWord>)paragraphVectors.getLookupTable();

            //Este loop percorre cada documento não rotulado no `unClassifiedIterator`

            while (testeIterator.hasNextDocument()) {

                //Para cada documento, seu conteúdo é tokenizado usando `tokenizerFactory`, e os tokens são armazenados em `documentAsTokens`.
                LabelledDocument unlabelledDocument = testeIterator.nextDocument();
                List<String> documentAsTokens = tokenizerFactory.create(unlabelledDocument.getContent()).getTokens();

                idsUnlabelledDocument.add(unlabelledDocument.getContent().length());

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

                List<String> labels = labelAwareIterator.getLabelsSource().getLabels();

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
                    mapTest.put(unlabelledDocument.getContent().length(), result.get(0).getFirst());
                } else {
                    mapTest.put(unlabelledDocument.getContent().length(), result.get(1).getFirst());
                }

            /*  SIMILARIDADE DO COSSENO
                A similaridade do cosseno é a similaridade entre dois vetores diferentes de zero medido pelo cosseno
            do ângulo entre eles. Essa métrica mede a orientação em vez da magnitude porque a similaridade
            de cosseno calcula o ângulo entre os vetores do documento em vez da contagem de palavras.
                Se o ângulo for zero, então o valor do cosseno chega a 1, indicando que são muito semelhantes.
            Se a semelhança do cosseno for próxima de zero, isso indica que há menos semelhança
            entre os documentos e os vetores do documento serão ortogonais (perpendiculares)
            entre si. Além disso, os documentos que são diferentes entre si produzirão uma similaridade de
            cosseno negativa. Para tais documentos, a similaridade do cosseno pode ir até -1, indicando um
            ângulo de 1.800 entre os vetores do documento.
            */

                //Por fim, imprime os rótulos e suas pontuações de similaridade correspondentes no log.
            /*for (Pair<String, Double> score: result) {
                log.info("        " + score.getFirst() + ": " + score.getSecond());
            }*/

            }

            for (int k = 0; k < mapTest.size(); k++){
                //Aqui preciso ver se há os documentos do mapTest
                //no mapTrain, e se forem correspondentes, comparar o label
                if(mapTrain.containsKey(idsUnlabelledDocument.get(k))){
                    String labelTrain = mapTrain.get(idsUnlabelledDocument.get(k));
                    String labelTest = mapTest.get(idsUnlabelledDocument.get(k));

                    if(labelTrain.equals("happy") && labelTest.equals("happy")) truePositives++;
                    else if(labelTrain.equals("happy") && labelTest.equals("sad")) falsePositives++;
                    else if(labelTrain.equals("sad") && labelTest.equals("sad")) trueNegatives++;
                    else if(labelTrain.equals("sad") && labelTest.equals("happy")) falseNegatives++;
                }
            }

            System.out.println("=====================MATRIZ DE CONFUSAO=====================");
            System.out.println("happy          sad");
            System.out.println(truePositives + "          " + falseNegatives);
            System.out.println(falsePositives + "          " + trueNegatives);
        }

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
            System.out.println("Arquivo criado com sucesso.");
        } catch (IOException e){
            e.printStackTrace();
        }
    }
}

//Essa é a explicação detalhada do código. Abrange a configuração, o treinamento e o uso de um modelo de vetor de parágrafo
//para classificação de texto. A parte principal está em como ele representa documentos e rótulos como vetores e mede sua
//similaridade usando similaridade de cosseno.
