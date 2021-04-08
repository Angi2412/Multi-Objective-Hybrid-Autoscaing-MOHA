/**
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
package tools.descartes.teastore.image.storage.rules;

import static org.junit.Assert.assertTrue;

import org.junit.Before;
import org.junit.Test;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import tools.descartes.teastore.image.StoreImage;
import tools.descartes.teastore.image.storage.rules.StoreAll;

public class TestStoreAll {

  @Mock
  private StoreImage mockedImg;

  @Before
  public void initialize() {
    MockitoAnnotations.initMocks(this);
  }

  @Test
  public void testRule() {
    StoreAll<StoreImage> uut = new StoreAll<>();
    assertTrue(uut.test(mockedImg));
    assertTrue(uut.test(null));
  }

}
