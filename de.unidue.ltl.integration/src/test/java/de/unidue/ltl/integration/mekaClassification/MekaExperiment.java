package de.unidue.ltl.integration.mekaClassification;

import static org.junit.Assert.assertEquals;

import org.dkpro.tc.core.Constants;
import org.dkpro.tc.evaluation.Id2Outcome;
import org.dkpro.tc.evaluation.evaluator.EvaluatorBase;
import org.dkpro.tc.evaluation.evaluator.EvaluatorFactory;
import org.dkpro.tc.evaluation.measures.label.MicroFScore;
import org.dkpro.tc.ml.weka.task.WekaTestTask;
import org.junit.Test;

import de.unidue.ltl.integration.ContextMemoryReport;
import de.unidue.ltl.integration.mekaMultiLabel.MekaReutersDemo;

public class MekaExperiment {
    @Test
    public void run()
        throws Exception
    {
        ContextMemoryReport.key = WekaTestTask.class.getName();
        new MekaReutersDemo().runTrainTest(MekaReutersDemo.getParameterSpace());

        Id2Outcome o = new Id2Outcome(ContextMemoryReport.id2outcome, Constants.LM_MULTI_LABEL);
        EvaluatorBase createEvaluator = EvaluatorFactory.createEvaluator(o, true, false);
        Double result = createEvaluator.calculateEvaluationMeasures()
                .get(MicroFScore.class.getSimpleName());
        assertEquals(0.357, result, 0.01);
    }
}
