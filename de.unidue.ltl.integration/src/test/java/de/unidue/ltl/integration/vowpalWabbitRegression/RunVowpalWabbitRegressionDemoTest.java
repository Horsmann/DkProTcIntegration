package de.unidue.ltl.integration.vowpalWabbitRegression;

import static org.junit.Assert.assertEquals;

import org.dkpro.tc.ml.report.util.Tc2LtlabEvalConverter;
import org.dkpro.tc.ml.vowpalwabbit.VowpalWabbitTestTask;
import org.junit.Test;

import de.unidue.ltl.evaluation.core.EvaluationData;
import de.unidue.ltl.evaluation.measures.correlation.SpearmanCorrelation;
import de.unidue.ltl.integration.ContextMemoryReport;

public class RunVowpalWabbitRegressionDemoTest
{
    @Test
    public void run()
        throws Exception
    {
        ContextMemoryReport.key = VowpalWabbitTestTask.class.getName();
        new VowpalWabbitRegressionDemo().run();

        EvaluationData<Double> d = Tc2LtlabEvalConverter.convertRegressionModeId2Outcome(ContextMemoryReport.id2outcome);
        SpearmanCorrelation a = new SpearmanCorrelation(d);
        
        assertEquals(0.831, a.getResult(), 0.1);
    }
}
