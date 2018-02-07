package de.unidue.ltl.integration.wekaRegression;

import static org.junit.Assert.assertEquals;

import org.dkpro.tc.ml.report.util.Tc2LtlabEvalConverter;
import org.dkpro.tc.ml.weka.task.WekaTestTask;
import org.junit.Test;

import de.unidue.ltl.evaluation.core.EvaluationData;
import de.unidue.ltl.evaluation.measures.correlation.SpearmanCorrelation;
import de.unidue.ltl.integration.ContextMemoryReport;

public class RunWekaRegressionDemoTest
{
    @Test
    public void run()
        throws Exception
    {
        ContextMemoryReport.key = WekaTestTask.class.getName();
        new WekaRegressionDemo().run();

        EvaluationData<Double> d = Tc2LtlabEvalConverter.convertRegressionModeId2Outcome(ContextMemoryReport.id2outcome);
        SpearmanCorrelation a = new SpearmanCorrelation(d);
        
        assertEquals(0.82345898, a.getResult(), 0.01);
    }
}
