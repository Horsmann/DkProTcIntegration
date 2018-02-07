package de.unidue.ltl.integration.mekaClassification;

import static org.junit.Assert.assertEquals;

import org.dkpro.tc.ml.report.util.Tc2LtlabEvalConverter;
import org.junit.Test;

import de.unidue.ltl.evaluation.core.EvaluationData;
import de.unidue.ltl.evaluation.measures.categorial.Fscore;
import de.unidue.ltl.integration.ContextMemoryReport;
import de.unidue.ltl.integration.mekaMultiLabel.MekaReutersDemo;

public class MekaExperiment {
    @Test
    public void run()
        throws Exception
    {
        new MekaReutersDemo().runTrainTest(MekaReutersDemo.getParameterSpace());

        EvaluationData<String> d = Tc2LtlabEvalConverter.convertMultiLabelModeId2Outcome(ContextMemoryReport.id2outcome);
        Fscore<String> f = new Fscore<>(d);
        assertEquals(0.357, f.getMicroFscore(), 0.01);
    }
}
