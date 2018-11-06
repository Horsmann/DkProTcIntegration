package de.unidue.ltl.integration.crf;

import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.uima.analysis_engine.AnalysisEngineDescription;
import org.apache.uima.collection.CollectionReaderDescription;
import org.apache.uima.fit.factory.CollectionReaderFactory;
import org.apache.uima.resource.ResourceInitializationException;
import org.dkpro.lab.Lab;
import org.dkpro.lab.task.BatchTask.ExecutionPolicy;
import org.dkpro.lab.task.Dimension;
import org.dkpro.lab.task.ParameterSpace;
import org.dkpro.tc.api.features.TcFeatureFactory;
import org.dkpro.tc.api.features.TcFeatureSet;
import org.dkpro.tc.core.Constants;
import org.dkpro.tc.features.maxnormalization.TokenLengthRatio;
import org.dkpro.tc.features.ngram.CharacterNGram;
import org.dkpro.tc.ml.crfsuite.CrfSuiteAdapter;
import org.dkpro.tc.ml.experiment.ExperimentTrainTest;

import de.tudarmstadt.ukp.dkpro.core.io.tei.TeiReader;
import de.unidue.ltl.integration.ContextMemoryReport;
import de.unidue.ltl.integration.SequenceOutcomeAnnotator;

public class PosTagging implements Constants
{
    public static void main(String[] args)
        throws Exception
    {
        new PosTagging().run();
    }

    public void run()
        throws Exception
    {
        // This is used to ensure that the required DKPRO_HOME environment variable is set.
        // Ensures that people can run the experiments even if they haven't read the setup
        // instructions first :)
        System.setProperty("DKPRO_HOME", "target/" + PosTagging.class.getSimpleName());
        
        Map<String, Object> config = new HashMap<>();
        config.put(DIM_CLASSIFICATION_ARGS, new Object[] { new CrfSuiteAdapter(), CrfSuiteAdapter.ALGORITHM_AVERAGED_PERCEPTRON });
        config.put(DIM_DATA_WRITER, new CrfSuiteAdapter().getDataWriterClass());
        config.put(DIM_FEATURE_USE_SPARSE, new CrfSuiteAdapter().useSparseFeatures());
        
        Dimension<Map<String, Object>> mlas = Dimension.createBundle("config", config);

        ParameterSpace pSpace = getParameterSpace(Constants.FM_SEQUENCE, Constants.LM_SINGLE_LABEL,
                mlas, null);

        PosTagging experiment = new PosTagging();
        experiment.runTrainTest(pSpace);
    }

    public static ParameterSpace getParameterSpace(String featureMode, String learningMode,
            Dimension<Map<String, Object>> mlas, Dimension<List<String>> dimFilters)
                throws ResourceInitializationException
    {
        // configure training and test data reader dimension
        Map<String, Object> dimReaders = new HashMap<String, Object>();

        CollectionReaderDescription train = CollectionReaderFactory.createReaderDescription(
                TeiReader.class, TeiReader.PARAM_LANGUAGE, "en", TeiReader.PARAM_SOURCE_LOCATION,
                "src/main/resources/posTagging/train", TeiReader.PARAM_PATTERNS, "*.xml");
        dimReaders.put(Constants.DIM_READER_TRAIN, train);

        CollectionReaderDescription test = CollectionReaderFactory.createReaderDescription(
                TeiReader.class, TeiReader.PARAM_LANGUAGE, "en", TeiReader.PARAM_SOURCE_LOCATION,
                "src/main/resources/posTagging/test", TeiReader.PARAM_PATTERNS, "*.xml");
        dimReaders.put(Constants.DIM_READER_TEST, test);

        Dimension<TcFeatureSet> dimFeatureSets = Dimension.create(Constants.DIM_FEATURE_SET,
                new TcFeatureSet(TcFeatureFactory.create(TokenLengthRatio.class),
                        TcFeatureFactory.create(CharacterNGram.class,
                                CharacterNGram.PARAM_NGRAM_MIN_N, 1,
                                CharacterNGram.PARAM_NGRAM_MAX_N, 1,
                                CharacterNGram.PARAM_NGRAM_USE_TOP_K, 50),
                        TcFeatureFactory.create(CharacterNGram.class,
                                CharacterNGram.PARAM_NGRAM_MIN_N, 2,
                                CharacterNGram.PARAM_NGRAM_MAX_N, 2,
                                CharacterNGram.PARAM_NGRAM_USE_TOP_K, 750),
                        TcFeatureFactory.create(CharacterNGram.class,
                                CharacterNGram.PARAM_NGRAM_MIN_N, 3,
                                CharacterNGram.PARAM_NGRAM_MAX_N, 3,
                                CharacterNGram.PARAM_NGRAM_USE_TOP_K, 750),
                        TcFeatureFactory.create(CharacterNGram.class,
                                CharacterNGram.PARAM_NGRAM_MIN_N, 4,
                                CharacterNGram.PARAM_NGRAM_MAX_N, 4,
                                CharacterNGram.PARAM_NGRAM_USE_TOP_K, 750)
                        ));

        ParameterSpace pSpace = new ParameterSpace(Dimension.createBundle("readers", dimReaders),
                Dimension.create(Constants.DIM_LEARNING_MODE, learningMode),
                Dimension.create(Constants.DIM_FEATURE_MODE, featureMode), dimFeatureSets,
                mlas);

        return pSpace;
    }

    protected AnalysisEngineDescription getPreprocessing()
        throws ResourceInitializationException
    {
        return createEngineDescription(SequenceOutcomeAnnotator.class);
    }

    public void runTrainTest(ParameterSpace pSpace)
        throws Exception
    {
        ExperimentTrainTest batch = new ExperimentTrainTest("BrownTrainTest");
        batch.setParameterSpace(pSpace);
        batch.setPreprocessing(getPreprocessing());
        batch.setExecutionPolicy(ExecutionPolicy.RUN_AGAIN);
        batch.addReport(ContextMemoryReport.class);

        Lab.getInstance().run(batch);
    }
}
