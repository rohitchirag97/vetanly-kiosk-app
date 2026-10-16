/**
 * Outcome of a single enrollment attempt.
 *
 * - `enrolled` means a new sample was stored.
 * - `redundant` means the face was valid but too similar to an existing sample.
 * - `noFace`, `multipleFaces`, `spoof`, and `lowQuality` are retryable capture
 *   failures.
 */
export type EnrollFaceStatus =
  | 'enrolled'
  | 'redundant'
  | 'noFace'
  | 'multipleFaces'
  | 'spoof'
  | 'lowQuality'

/**
 * Result from enrolling a frame. `enrollment` contains the full serialized
 * template set for the identity when native enrollment state exists.
 */
export interface EnrollFaceResult {
  /** Outcome of the enrollment attempt. */
  status: EnrollFaceStatus
  /**
   * Binary enrollment payload for the identity.
   *
   * Persist this value when present so the identity can be restored with
   * `FaceRecognizer.addEnrollment` in a later app session.
   */
  enrollment?: ArrayBuffer
}
