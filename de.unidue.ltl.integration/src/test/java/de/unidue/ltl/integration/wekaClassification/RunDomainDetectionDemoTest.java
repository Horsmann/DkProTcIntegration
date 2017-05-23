package de.unidue.ltl.integration.wekaClassification;

import static org.junit.Assert.assertEquals;

import org.dkpro.tc.core.Constants;
import org.dkpro.tc.evaluation.Id2Outcome;
import org.dkpro.tc.evaluation.evaluator.EvaluatorBase;
import org.dkpro.tc.evaluation.evaluator.EvaluatorFactory;
import org.dkpro.tc.evaluation.measures.label.Accuracy;
import org.dkpro.tc.ml.weka.task.WekaTestTask;
import org.junit.Test;

import de.unidue.ltl.integration.ContextMemoryReport;

public class RunDomainDetectionDemoTest
{
    @Test
    public void run()
        throws Exception
    {
        ContextMemoryReport.key = WekaTestTask.class.getName();
        new WekaTwentyNewsgroupsDemo().run();

        Id2Outcome o = new Id2Outcome(ContextMemoryReport.id2outcome, Constants.LM_SINGLE_LABEL);
        EvaluatorBase createEvaluator = EvaluatorFactory.createEvaluator(o, true, false);
        Double result = createEvaluator.calculateEvaluationMeasures()
                .get(Accuracy.class.getSimpleName());
        assertEquals(0.75118, result, 0.01);
    }
}
