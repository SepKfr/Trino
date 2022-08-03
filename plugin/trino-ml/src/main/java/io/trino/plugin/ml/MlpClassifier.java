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

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import io.trino.plugin.ml.type.ModelType;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static io.trino.plugin.ml.type.ClassifierType.BIGINT_CLASSIFIER;
import static java.util.Objects.requireNonNull;

public class MlpClassifier
        extends AbstractMlpModel
        implements Classifier<Integer>
{
    public MlpClassifier()
    {
        this(MlpUtils.parseParameters(""));
    }

    protected MlpClassifier(MlpParameter params)
    {
        super(params);
    }

    private MlpClassifier(Model mlp)
    {
        super(mlp);
    }

    public MlpClassifier deserialize(byte[] modelData)
    {
        // TODO do something with the hyperparameters
        try {
            // UTF-8 should work, though the only thing we can say about the charset is that it's guaranteed to be 8-bits
            model.load(new ByteArrayInputStream(modelData));
            return new MlpClassifier(model);
        }
        catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        catch (MalformedModelException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public Integer classify(FeatureVector features)
    {
        requireNonNull(model, "model is null");
        Translator<FeatureVector, Classifications> translator = new Translator<FeatureVector, Classifications>()
        {
            @Override
            public Classifications processOutput(TranslatorContext ctx, NDList list)
            {
                NDArray probabilities = list.singletonOrThrow().softmax(0);
                List<String> classNames = IntStream.range(0, 10).mapToObj(String::valueOf).collect(Collectors.toList());
                return new Classifications(classNames, probabilities);
            }

            @Override
            public NDList processInput(TranslatorContext ctx, FeatureVector input)
            {
                NDArray array = ctx.getNDManager().zeros(new Shape(input.size()));
                for (int i = 0; i < input.size(); i++) {
                    array.add(input.getFeatures().get(i));
                }
                return new NDList(array);
            }
        };
        try (Predictor<FeatureVector, Classifications> predictor = model.newPredictor(translator)) {
            var classifications = predictor.predict(features);
            return Integer.parseInt(classifications.best().getClassName());
        }
        catch (TranslateException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public ModelType getType()
    {
        return BIGINT_CLASSIFIER;
    }
}
