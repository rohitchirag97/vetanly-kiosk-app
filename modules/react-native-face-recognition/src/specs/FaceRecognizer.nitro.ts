import type { HybridObject } from 'react-native-nitro-modules'
import type { Frame } from 'react-native-vision-camera'
import type { EnrollFaceResult } from '../types/EnrollFaceResult'
import type { NativeRecognizedFace } from '../types/NativeRecognizedFace'

/**
 * Processes VisionCamera frames using a ready native face-recognition engine.
 *
 * The caller retains ownership of every frame and must dispose it after the
 * synchronous operation returns.
 *
 * @see {@linkcode FaceRecognizerFactory.create}
 */
export interface FaceRecognizer
  extends HybridObject<{
    ios: 'c++'
    android: 'c++'
  }> {
  /**
   * Recognizes every usable face in a frame.
   *
   * The native pipeline owns detection cadence, tracking, embedding cadence,
   * liveness checks, and match reuse. `matchAge` is measured in processed
   * frames; `0` means the match was computed for the current frame.
   */
  recognizeFaces(frame: Frame): NativeRecognizedFace[]
  /**
   * Detects exactly one face and adds its embedding to the sample set stored
   * under `id`.
   *
   * Each identity keeps several samples so it can be recognized across pose,
   * expression, and lighting changes. Call this repeatedly while guiding the
   * user through a few angles. Frames whose embedding is nearly identical to an
   * existing sample are rejected (`status: 'redundant'`) so the set stays
   * diverse, and the set is capped once full. Persist
   * {@linkcode EnrollFaceResult.enrollment} when present to restore the identity
   * later via {@linkcode addEnrollment}.
   */
  enrollFace(id: string, frame: Frame): EnrollFaceResult

  /**
   * Restores an identity from a previously persisted enrollment buffer,
   * replacing any samples currently stored under `id`.
   */
  addEnrollment(id: string, enrollment: ArrayBuffer): void

  /**
   * Returns the serialized enrollment buffer for `id`.
   *
   * Throws when the identity is unknown.
   */
  getEnrollment(id: string): ArrayBuffer

  /** Returns the ids of every enrolled identity. */
  enrollmentIds(): string[]

  /** Removes a single enrolled identity. No-op if `id` is unknown. */
  removeEnrollment(id: string): void

  /** Removes every enrollment from the native runtime matching index. */
  clearEnrollments(): void
}
