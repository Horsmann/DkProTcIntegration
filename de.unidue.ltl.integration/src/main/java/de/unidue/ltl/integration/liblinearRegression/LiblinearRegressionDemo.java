package de.unidue.ltl.integration.liblinearRegression;

import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;

import java.util.HashMap;
import java.util.Map;

import org.apache.uima.analysis_engine.AnalysisEngineDescription;
import org.apache.uima.collection.CollectionReaderDescription;
import org.apache.uima.fit.factory.CollectionReaderFactory;
import org.apache.uima.resource.ResourceInitializationException;
import org.dkpro.lab.Lab;
import org.dkpro.lab.task.Dimension;
import org.dkpro.lab.task.ParameterSpace;
import org.dkpro.tc.api.features.TcFeatureFactory;
import org.dkpro.tc.api.features.TcFeatureSet;
import org.dkpro.tc.core.Constants;
import org.dkpro.tc.features.maxnormalization.SentenceRatioPerDocument;
import org.dkpro.tc.features.maxnormalization.TokenRatioPerDocument;
import org.dkpro.tc.ml.experiment.ExperimentTrainTest;
import org.dkpro.tc.ml.liblinear.LiblinearAdapter;

import de.tudarmstadt.ukp.dkpro.core.tokit.BreakIteratorSegmenter;
import de.unidue.ltl.integration.ContextMemoryReport;
import de.unidue.ltl.integration.wekaRegression.EssayScoreReader;

public class LiblinearRegressionDemo
    implements Constants
{

    public static void main(String[] args)
        throws Exception
    {
        new LiblinearRegressionDemo().run();
    }

    public void run()
        throws Exception
    {
        System.setProperty("DKPRO_HOME", "target/" + LiblinearRegressionDemo.class.getSimpleName());
        ParameterSpace pSpace = getParameterSpace();

        LiblinearRegressionDemo experiment = new LiblinearRegressionDemo();
        experiment.runTrainTest(pSpace);
    }

    public static ParameterSpace getParameterSpace()
        throws ResourceInitializationException
    {
        // configure training and test data reader dimension
        // train/test will use both, while cross-validation will only use the train part
        // The reader is also responsible for setting the labels/outcome on all
        // documents/instances it creates.
        Map<String, Object> dimReaders = new HashMap<String, Object>();

        CollectionReaderDescription readerTrain = CollectionReaderFactory.createReaderDescription(
                EssayScoreReader.class, EssayScoreReader.PARAM_SOURCE_LOCATION,
                "src/main/resources/essays/train.txt", EssayScoreReader.PARAM_LANGUAGE, "en");
        dimReaders.put(DIM_READER_TRAIN, readerTrain);

        CollectionReaderDescription readerTest = CollectionReaderFactory.createReaderDescription(
                EssayScoreReader.class, EssayScoreReader.PARAM_SOURCE_LOCATION,
                "src/main/resources/essays/test.txt", EssayScoreReader.PARAM_LANGUAGE, "en");
        dimReaders.put(DIM_READER_TEST, readerTest);
        

        Map<String, Object> config = new HashMap<>();
        config.put(DIM_CLASSIFICATION_ARGS, new Object[] { new LiblinearAdapter(),  "-s", "6" });
        config.put(DIM_DATA_WRITER, new LiblinearAdapter().getDataWriterClass());
        config.put(DIM_FEATURE_USE_SPARSE, new LiblinearAdapter().useSparseFeatures());
        
        Dimension<Map<String, Object>> mlas = Dimension.createBundle("config", config);

        Dimension<TcFeatureSet> dimFeatureSets = Dimension.create(DIM_FEATURE_SET,
                new TcFeatureSet(TcFeatureFactory.create(TokenRatioPerDocument.class),
                        TcFeatureFactory.create(SentenceRatioPerDocument.class)));

        ParameterSpace pSpace = new ParameterSpace(Dimension.createBundle("readers", dimReaders),
                Dimension.create(DIM_LEARNING_MODE, LM_REGRESSION),
                Dimension.create(DIM_FEATURE_MODE, FM_DOCUMENT), dimFeatureSets,
                mlas);

        return pSpace;
    }

    // ##### TRAIN-TEST #####
    protected void runTrainTest(ParameterSpace pSpace)
        throws Exception
    {
        ExperimentTrainTest batch = new ExperimentTrainTest("LiblinearRegressionDemo");
        batch.setPreprocessing(getPreprocessing());
        batch.setParameterSpace(pSpace);
        batch.addReport(ContextMemoryReport.class);

        // Run
        Lab.getInstance().run(batch);
    }

    protected AnalysisEngineDescription getPreprocessing()
        throws ResourceInitializationException
    {
        return createEngineDescription(createEngineDescription(BreakIteratorSegmenter.class));
    }
}
