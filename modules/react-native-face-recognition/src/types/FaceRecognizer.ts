import type { Frame } from 'react-native-vision-camera'
import type { FaceRecognizer as NativeFaceRecognizer } from '../specs/FaceRecognizer.nitro'
import type { RecognizedFace } from './RecognizedFace'

/**
 * Public recognizer API with status-aware face results.
 *
 * This wraps the generated Nitro type so `recognizeFaces` returns the
 * discriminated `RecognizedFace` union exposed to JavaScript consumers while
 * preserving the rest of the native recognizer methods.
 */
export type FaceRecognizer = Omit<NativeFaceRecognizer, 'recognizeFaces'> & {
  /**
   * Synchronously detects and recognizes faces in a VisionCamera frame.
   *
   * The caller still owns the frame and must dispose it after processing. The
   * native implementation may reuse tracking and match results across frames
   * according to the configured cadence options.
   */
  recognizeFaces(frame: Frame): RecognizedFace[]
}
