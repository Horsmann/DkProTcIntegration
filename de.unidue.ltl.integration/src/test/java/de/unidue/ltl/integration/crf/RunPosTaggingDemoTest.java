package de.unidue.ltl.integration.crf;

import static org.junit.Assert.*;

import org.dkpro.tc.core.Constants;
import org.dkpro.tc.evaluation.Id2Outcome;
import org.dkpro.tc.evaluation.evaluator.EvaluatorBase;
import org.dkpro.tc.evaluation.evaluator.EvaluatorFactory;
import org.dkpro.tc.evaluation.measures.label.Accuracy;
import org.dkpro.tc.ml.crfsuite.task.CRFSuiteTestTask;
import org.junit.Test;

import de.unidue.ltl.integration.ContextMemoryReport;

public class RunPosTaggingDemoTest
{
    @Test
    public void run()
        throws Exception
    {
        ContextMemoryReport.key = CRFSuiteTestTask.class.getName();
        new PosTagging().run();

        Id2Outcome o = new Id2Outcome(ContextMemoryReport.id2outcome, Constants.LM_SINGLE_LABEL);
        EvaluatorBase createEvaluator = EvaluatorFactory.createEvaluator(o, true, false);
        Double result = createEvaluator.calculateEvaluationMeasures()
                .get(Accuracy.class.getSimpleName());
        assertEquals(0.92177, result, 0.01);
    }
}
