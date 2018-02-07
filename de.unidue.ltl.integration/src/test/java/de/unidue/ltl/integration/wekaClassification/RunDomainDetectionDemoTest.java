package de.unidue.ltl.integration.wekaClassification;

import static org.junit.Assert.assertEquals;

import org.dkpro.tc.ml.report.util.Tc2LtlabEvalConverter;
import org.dkpro.tc.ml.weka.task.WekaTestTask;
import org.junit.Test;

import de.unidue.ltl.evaluation.core.EvaluationData;
import de.unidue.ltl.evaluation.measures.Accuracy;
import de.unidue.ltl.integration.ContextMemoryReport;

public class RunDomainDetectionDemoTest
{
    @Test
    public void run()
        throws Exception
    {
        ContextMemoryReport.key = WekaTestTask.class.getName();
        new WekaTwentyNewsgroupsDemo().run();


        EvaluationData<String> d = Tc2LtlabEvalConverter.convertSingleLabelModeId2Outcome(ContextMemoryReport.id2outcome);
        Accuracy<String> a = new Accuracy<>(d);

        assertEquals(0.75118, a.getResult(), 0.01);
    }
}
