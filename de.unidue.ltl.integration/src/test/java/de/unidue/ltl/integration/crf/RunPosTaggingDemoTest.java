package de.unidue.ltl.integration.crf;

import static org.junit.Assert.assertEquals;

import org.dkpro.tc.core.Constants;
import org.dkpro.tc.evaluation.Id2Outcome;
import org.dkpro.tc.evaluation.evaluator.EvaluatorBase;
import org.dkpro.tc.evaluation.evaluator.EvaluatorFactory;
import org.dkpro.tc.evaluation.measures.label.Accuracy;
import org.junit.Test;

import de.unidue.ltl.integration.ContextMemoryReport;

public class RunPosTaggingDemoTest
{
    @Test
    public void run()
        throws Exception
    {
        new PosTagging().run();
        ContextMemoryReport.key = PosTagging.class.getName();

        Id2Outcome o = new Id2Outcome(ContextMemoryReport.id2outcome, Constants.LM_SINGLE_LABEL);
        EvaluatorBase createEvaluator = EvaluatorFactory.createEvaluator(o, true, false);
        Double result = createEvaluator.calculateEvaluationMeasures()
                .get(Accuracy.class.getSimpleName());
        assertEquals(0.75, result, 0.0001);
    }
}
