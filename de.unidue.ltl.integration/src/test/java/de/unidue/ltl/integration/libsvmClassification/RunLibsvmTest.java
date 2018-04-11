package de.unidue.ltl.integration.libsvmClassification;

import static org.junit.Assert.assertEquals;

import org.dkpro.tc.ml.libsvm.LibsvmTestTask;
import org.dkpro.tc.ml.report.util.Tc2LtlabEvalConverter;
import org.junit.Test;

import de.unidue.ltl.evaluation.core.EvaluationData;
import de.unidue.ltl.evaluation.measures.Accuracy;
import de.unidue.ltl.integration.ContextMemoryReport;

public class RunLibsvmTest
{
    @Test
    public void run()
        throws Exception
    {
        ContextMemoryReport.key = LibsvmTestTask.class.getName();
        new LibsvmNewsgroupsDemo().run();

        EvaluationData<String> d = Tc2LtlabEvalConverter.convertSingleLabelModeId2Outcome(ContextMemoryReport.id2outcome);
        Accuracy<String> a = new Accuracy<>(d);
        
        assertEquals(0.58, a.getResult(), 0.01);
    }
}
