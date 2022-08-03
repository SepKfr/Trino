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

import static io.trino.plugin.ml.type.RegressorType.REGRESSOR;
import static java.util.Objects.requireNonNull;

public class MlpRegressor
        extends AbstractMlpModel
        implements Regressor
{
    public MlpRegressor(MlpParameter params)
    {
        super(params);
    }

    public MlpRegressor()
    {
        this(MlpUtils.parseParameters(""));
    }

    private MlpRegressor(Model mlp)
    {
        super(mlp);
    }

    @Override
    public ModelType getType()
    {
        return REGRESSOR;
    }

    public MlpRegressor deserialize(byte[] modelData)
    {
        // TODO do something with the hyperparameters
        try {
            // UTF-8 should work, though the only thing we can say about the charset is that it's guaranteed to be 8-bits
            model.load(new ByteArrayInputStream(modelData));
            return new MlpRegressor(model);
        }
        catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        catch (MalformedModelException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public double regress(FeatureVector features)
    {
        requireNonNull(model, "model is null");
        Translator<FeatureVector, Double> translator = new Translator<FeatureVector, Double>()
        {
            @Override
            public Double processOutput(TranslatorContext ctx, NDList list) throws Exception
            {
                return list.get(0).getDouble();
            }

            @Override
            public NDList processInput(TranslatorContext ctx, FeatureVector input) throws Exception
            {
                NDArray array = ctx.getNDManager().zeros(new Shape(input.size()));
                for (int i = 0; i < input.size(); i++) {
                    array.add(input.getFeatures().get(i));
                }
                return new NDList(array);
            }
        };
        try (Predictor<FeatureVector, Double> predictor = model.newPredictor(translator)) {
            var regression = predictor.predict(features);
            return regression;
        }
        catch (TranslateException e) {
            throw new RuntimeException(e);
        }
    }
}
