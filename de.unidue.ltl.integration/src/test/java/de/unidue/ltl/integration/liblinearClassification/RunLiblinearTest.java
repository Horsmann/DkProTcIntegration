package de.unidue.ltl.integration.liblinearClassification;

import static org.junit.Assert.assertEquals;

import org.dkpro.tc.ml.libsvm.LibsvmTestTask;
import org.dkpro.tc.ml.report.util.Tc2LtlabEvalConverter;
import org.junit.Test;

import de.unidue.ltl.evaluation.core.EvaluationData;
import de.unidue.ltl.evaluation.measures.Accuracy;
import de.unidue.ltl.integration.ContextMemoryReport;

public class RunLiblinearTest
{
    @Test
    public void run()
        throws Exception
    {
        ContextMemoryReport.key = LibsvmTestTask.class.getName();
        new LiblinearNewsgroupsDemo().run();

        EvaluationData<String> d = Tc2LtlabEvalConverter.convertSingleLabelModeId2Outcome(ContextMemoryReport.id2outcome);
        Accuracy<String> a = new Accuracy<>(d);
        
        assertEquals(0.79414, a.getResult(), 0.01);
    }
}
