/**
 * Copyright 2017
 * Ubiquitous Knowledge Processing (UKP) Lab
 * Technische Universität Darmstadt
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see http://www.gnu.org/licenses/.
 */
package de.unidue.ltl.integration;

import java.io.File;
import java.util.Set;

import org.dkpro.lab.storage.StorageService;
import org.dkpro.tc.core.Constants;
import org.dkpro.tc.core.task.TcTaskTypeUtil;
import org.dkpro.tc.ml.report.TcAbstractReport;

/**
 * This is a slightly ugly solution for recording the DKPro Lab output folder of an experiment to
 * read result files in JUnit tests
 */
public class ContextMemoryReport
    extends TcAbstractReport
{

    /**
     * Name of the folder which will contain the id2outcome.txt that shall be used for evaluation
     * for TrainTest scenarios this is the *TestTask class of the respective machine learning
     * adapter e.g. Weka for CrossValidation experiments it is the ExperimentCrossValidation folder
     */
    public static String key; // this has to be set BEFORE the pipeline runs

    public static File id2outcome;

	@Override
	public void execute() throws Exception {
		
		StorageService storageService = getContext().getStorageService();
		
		Set<String> taskIds = collectTasks(getTaskIdsFromMetaData(getSubtasks()));
		
		for (String id : taskIds) {
			if (TcTaskTypeUtil.isMachineLearningAdapterTask(storageService, id)) {
				id2outcome = storageService.locateKey(id, Constants.ID_OUTCOME_KEY);
				return;
			}
			if (TcTaskTypeUtil.isCrossValidationTask(storageService, id)) {
				id2outcome = storageService.locateKey(id, Constants.FILE_COMBINED_ID_OUTCOME_KEY);
				return;
			}
		}
	}
}
