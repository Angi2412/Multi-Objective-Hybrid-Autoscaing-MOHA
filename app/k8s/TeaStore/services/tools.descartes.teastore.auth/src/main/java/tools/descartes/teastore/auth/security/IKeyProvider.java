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

package tools.descartes.teastore.auth.security;

import tools.descartes.teastore.entities.message.SessionBlob;

/**
 * Provides keys for the security provider. The key provider must ensure that
 * keys accross replicated stores are consistent.
 * 
 * @author Joakim von Kistowski
 *
 */
public interface IKeyProvider {

  /**
   * Returns a key for a session blob. Key must be the same, regardless of the
   * store instance upon which this call is made.
   * 
   * @param blob
   *          The blob to secure.
   * @return The key.
   */
  public String getKey(SessionBlob blob);

}
