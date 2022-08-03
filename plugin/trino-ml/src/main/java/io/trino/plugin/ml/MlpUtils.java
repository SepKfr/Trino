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

import ai.djl.engine.Engine;
import com.google.common.base.Splitter;
import io.trino.spi.TrinoException;

import static com.google.common.base.Preconditions.checkArgument;
import static io.trino.spi.StandardErrorCode.INVALID_FUNCTION_ARGUMENT;
import static java.lang.String.format;

public class MlpUtils
{
    private MlpUtils() {}

    public static MlpParameter parseParameters(String paramString)
    {
        MlpParameter params = new MlpParameter();
        params.maxGpus = Engine.getInstance().getGpuCount();
        params.batchSize = params.maxGpus > 0 ? 32 * params.maxGpus : 32;
        params.outputDir = "build/model";
        params.limit = Long.MAX_VALUE;
        params.modelDir = null;
        params.epoch = 2;
        params.isSymbolic = true;
        params.preTrained = false;

        for (String split : Splitter.on(',').omitEmptyStrings().split(paramString)) {
            String[] pair = split.split("=");
            checkArgument(pair.length == 2, "Invalid hyperparameters string for libsvm");
            String value = pair[1].trim();

            String key = pair[0].trim();
            switch (key) {
                case "batch_size":
                    params.batchSize = Integer.parseInt(value);
                    break;
                case "model_dir":
                    params.modelDir = value;
                    break;
                case "epoch":
                    params.epoch = Integer.parseInt(value);
                    break;
                default:
                    throw new TrinoException(INVALID_FUNCTION_ARGUMENT, format("Unknown parameter %s", pair[0]));
            }
        }
        return params;
    }
}
