/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.trino.plugin.ml;

import ai.djl.Model;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.engine.Engine;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.translate.TranslateException;
import com.google.common.util.concurrent.SimpleTimeLimiter;
import com.google.common.util.concurrent.TimeLimiter;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.SortedMap;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;

import static com.google.common.base.Throwables.throwIfUnchecked;
import static io.airlift.concurrent.Threads.threadsNamed;
import static java.util.Objects.requireNonNull;
import static java.util.concurrent.Executors.newCachedThreadPool;

public abstract class AbstractMlpModel
        implements io.trino.plugin.ml.Model
{
    protected Model model;

    protected MlpParameter params;

    protected AbstractMlpModel(MlpParameter params)
    {
        this.params = requireNonNull(params, "args are not null");
    }

    protected AbstractMlpModel(Model mlp)
    {
        this.model = requireNonNull(mlp, "model is null");
    }

    public byte[] getSerializedData()
    {
        try {
            model.save(Paths.get("."), "mlp");
            return Files.readAllBytes(model.getModelPath());
        }
        catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    public void train(Dataset dataset)
    {
        ExecutorService service = newCachedThreadPool(threadsNamed("mlp-trainer-" + System.identityHashCode(this) + "-%s"));
        try {
            TimeLimiter limiter = SimpleTimeLimiter.create(service);
            //TODO: this time limit should be configurable
            model = limiter.callWithTimeout(getTrainingFunction(params, dataset), 1, TimeUnit.HOURS);
        }
        catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(e);
        }
        catch (ExecutionException e) {
            Throwable cause = e.getCause();
            if (cause != null) {
                throwIfUnchecked(cause);
                throw new RuntimeException(cause);
            }
        }
        catch (Exception e) {
            throwIfUnchecked(e);
            throw new RuntimeException(e);
        }
        finally {
            service.shutdownNow();
        }
    }

    private static Callable<Model> getTrainingFunction(MlpParameter params, Dataset dataset) throws TranslateException, IOException
    {
        DefaultTrainingConfig config = setupTrainingConfig(params);

        int numFeatures = dataset.getDatapoints().get(0).size();

        Mlp block = new Mlp(numFeatures, 1, new int[]{128, 64});
        try (Model model = Model.newInstance("mlp")) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                List<Double> labels = dataset.getLabels();
                try (NDManager manager = NDManager.newBaseManager()) {
                    NDArray features = manager.zeros(new Shape(labels.size(), numFeatures));
                    NDArray lables = manager.zeros(new Shape(labels.size()));
                    for (int i = 0; i < dataset.getDatapoints().size(); i++) {
                        for (SortedMap.Entry<Integer, Double> feature : dataset.getDatapoints().get(i).getFeatures().entrySet()) {
                            features.add(feature.getValue());
                        }
                    }
                    for (Double label : labels) {
                        lables.add(label);
                    }
                    ArrayDataset data = new ArrayDataset.Builder()
                            .setData(features) // Set the Features
                            .optLabels(lables) // Set the Labels
                            .setSampling(1, false) // set the batch size and random sampling to false
                            .build();
                    int trainLen = (int) (0.8 * labels.size());
                    RandomAccessDataset trainingSet = data.subDataset(0, trainLen);
                    RandomAccessDataset validateSet = data.subDataset(trainLen, labels.size());

                    trainer.setMetrics(new Metrics());
                    Shape inputShape = new Shape(1, numFeatures);

                    trainer.initialize(inputShape);

                    EasyTrain.fit(trainer, params.epoch, trainingSet, validateSet);

                    return trainer::getModel;
                }
            }
        }
    }

    private static DefaultTrainingConfig setupTrainingConfig(MlpParameter params)
    {
        String outputDir = params.outputDir;
        SaveModelTrainingListener listener = new SaveModelTrainingListener(outputDir);
        listener.setSaveModelCallback(
                trainer -> {
                    TrainingResult result = trainer.getTrainingResult();
                    Model model = trainer.getModel();
                    float accuracy = result.getValidateEvaluation("Accuracy");
                    model.setProperty("Accuracy", String.format("%.5f", accuracy));
                    model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));
                });
        return new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                .addEvaluator(new Accuracy())
                .optDevices(Engine.getInstance().getDevices(params.maxGpus))
                .addTrainingListeners(TrainingListener.Defaults.logging(outputDir))
                .addTrainingListeners(listener);
    }
}
