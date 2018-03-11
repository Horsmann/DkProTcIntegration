package de.unidue.ltl.integration.liblinearRegression;

import static org.junit.Assert.assertEquals;

import org.dkpro.tc.ml.liblinear.LiblinearTestTask;
import org.dkpro.tc.ml.report.util.Tc2LtlabEvalConverter;
import org.junit.Test;

import de.unidue.ltl.evaluation.core.EvaluationData;
import de.unidue.ltl.evaluation.measures.correlation.SpearmanCorrelation;
import de.unidue.ltl.integration.ContextMemoryReport;

public class RunLiblinearRegressionDemoTest
{
    @Test
    public void run()
        throws Exception
    {
        ContextMemoryReport.key = LiblinearTestTask.class.getName();
        new LiblinearRegressionDemo().run();

        EvaluationData<Double> d = Tc2LtlabEvalConverter.convertRegressionModeId2Outcome(ContextMemoryReport.id2outcome);
        SpearmanCorrelation a = new SpearmanCorrelation(d);
        
        assertEquals(0.75, a.getResult(), 0.01);
    }
}
