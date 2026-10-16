import { NitroModules } from 'react-native-nitro-modules'
import type { FaceRecognizerFactory } from './specs/FaceRecognizerFactory.nitro'
import type { FaceRecognizer } from './types/FaceRecognizer'
import type { FaceRecognizerOptions } from './types/FaceRecognizerOptions'

const NativeFaceRecognizerFactory =
  NitroModules.createHybridObject<FaceRecognizerFactory>(
    'FaceRecognizerFactory'
  )

/**
 * Entry point for creating native face-recognition engines.
 *
 * Each recognizer owns native model state, tracking state, and its in-memory
 * enrollment index. Create one recognizer for an active camera session and
 * dispose it when the session ends.
 */
export const FaceRecognition = {
  /**
   * Loads the bundled native detector and recognizer models, then returns a
   * ready engine.
   *
   * Model loading is asynchronous, but frame processing on the returned
   * recognizer is synchronous so it can run from VisionCamera frame processors.
   * Rejects when native initialization fails, for example when a requested
   * execution provider is unavailable.
   */
  create(options: FaceRecognizerOptions): Promise<FaceRecognizer> {
    return NativeFaceRecognizerFactory.create(
      options
    ) as Promise<FaceRecognizer>
  },
}
