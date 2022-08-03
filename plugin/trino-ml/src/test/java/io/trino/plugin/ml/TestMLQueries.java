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

import com.google.common.collect.ImmutableMap;
import io.trino.Session;
import io.trino.plugin.tpch.TpchConnectorFactory;
import io.trino.testing.AbstractTestQueryFramework;
import io.trino.testing.LocalQueryRunner;
import io.trino.testing.QueryRunner;
import org.testng.annotations.Test;

import static io.trino.plugin.tpch.TpchMetadata.TINY_SCHEMA_NAME;
import static io.trino.testing.TestingSession.testSessionBuilder;

public class TestMLQueries
        extends AbstractTestQueryFramework
{
    @Override
    protected QueryRunner createQueryRunner()
    {
        Session defaultSession = testSessionBuilder()
                .setCatalog("local")
                .setSchema(TINY_SCHEMA_NAME)
                .build();

        LocalQueryRunner localQueryRunner = LocalQueryRunner.create(defaultSession);

        // add the tpch catalog
        // local queries run directly against the generator
        localQueryRunner.createCatalog(
                defaultSession.getCatalog().get(),
                new TpchConnectorFactory(1),
                ImmutableMap.of());

        localQueryRunner.installPlugin(new MLPlugin());
        return localQueryRunner;
    }

    @Test
    public void testPrediction()
    {
        assertQuery("SELECT classify(features(1, 2), model) " +
                "FROM (SELECT learn_classifier(labels, features) AS model FROM (VALUES (1, features(1, 2))) t(labels, features)) t2", "SELECT 1");
    }

    @Test
    public void testPredictionMlp()
    {
        assertQuery("SELECT classify(features(1, 2), model) " +
                "FROM (SELECT learn_mlp_classifier(labels, features) AS model FROM (VALUES (1, features(1, 2))) t(labels, features)) t2", "SELECT 1");
    }

    @Test
    public void testVarcharPrediction()
    {
        assertQuery("SELECT classify(features(1, 2), model) " +
                "FROM (SELECT learn_classifier(labels, features) AS model FROM (VALUES ('cat', features(1, 2))) t(labels, features)) t2", "SELECT 'cat'");
    }

    @Test
    public void testVarcharPredictionMlp()
    {
        assertQuery("SELECT classify(features(1, 2), model) " +
                "FROM (SELECT learn_mlp_classifier(labels, features) AS model FROM (VALUES ('cat', features(1, 2))) t(labels, features)) t2", "SELECT 'cat'");
    }
}
