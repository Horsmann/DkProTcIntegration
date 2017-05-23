package de.unidue.ltl.integration.liblinearRegression;

import static org.junit.Assert.assertEquals;

import org.dkpro.tc.core.Constants;
import org.dkpro.tc.evaluation.Id2Outcome;
import org.dkpro.tc.evaluation.evaluator.EvaluatorBase;
import org.dkpro.tc.evaluation.evaluator.EvaluatorFactory;
import org.dkpro.tc.evaluation.measures.regression.SpearmanCorrelation;
import org.dkpro.tc.ml.liblinear.LiblinearTestTask;
import org.dkpro.tc.ml.weka.task.WekaTestTask;
import org.junit.Test;

import de.unidue.ltl.integration.ContextMemoryReport;

public class RunLiblinearRegressionDemoTest
{
    @Test
    public void run()
        throws Exception
    {
        ContextMemoryReport.key = LiblinearTestTask.class.getName();
        new LiblinearRegressionDemo().run();

        Id2Outcome o = new Id2Outcome(ContextMemoryReport.id2outcome, Constants.LM_REGRESSION);
        EvaluatorBase createEvaluator = EvaluatorFactory.createEvaluator(o, true, false);
        Double result = createEvaluator.calculateEvaluationMeasures()
                .get(SpearmanCorrelation.class.getSimpleName());
        assertEquals(0.7731957, result, 0.01);
    }
}
