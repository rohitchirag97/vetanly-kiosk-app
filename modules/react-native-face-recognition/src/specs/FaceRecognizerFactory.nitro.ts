import type { HybridObject } from 'react-native-nitro-modules'
import type { FaceRecognizer } from './FaceRecognizer.nitro'
import type { FaceRecognizerOptions } from '../types/FaceRecognizerOptions'

/**
 * Nitro factory responsible for creating native recognizer instances.
 *
 * The factory itself is lightweight. Each recognizer returned from
 * {@linkcode create} owns loaded model sessions and mutable enrollment state.
 */
export interface FaceRecognizerFactory
  extends HybridObject<{ ios: 'c++'; android: 'c++' }> {
  /**
   * Initializes a native recognizer using the bundled model assets and supplied
   * runtime options.
   *
   * Rejects when model loading, provider selection, or native session creation
   * fails.
   */
  create(options: FaceRecognizerOptions): Promise<FaceRecognizer>
}
