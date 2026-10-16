import type { Point } from './Point'
import type { Rect } from './Rect'
import type { FaceRecognitionStatus } from './RecognizedFace'

/**
 * Raw result shape returned by the Nitro layer before TypeScript narrows it
 * into the public `RecognizedFace` discriminated union.
 */
export interface NativeRecognizedFace {
  /** Face bounds in oriented upright-display pixels. */
  bounds: Rect
  /** Detector confidence from `0` to `1`. */
  confidence: number
  /** Facial landmarks in oriented upright-display pixels. */
  landmarks: Point[]
  /** Recognition outcome for this face. */
  status: FaceRecognitionStatus
  /** Fresh embeddings collected so far. Present only while collecting. */
  samplesCollected?: number
  /** Fresh embeddings required. Present only while collecting. */
  samplesRequired?: number
  /** Matched enrollment id. Present only when `status` is `matched`. */
  enrollmentId?: string
  /** Cosine similarity for the matched identity. Present only for matches. */
  similarity?: number
  /** Processed-frame age of the reused match. `0` means current-frame match. */
  matchAge?: number
  /** Passive liveness probability. Present only when `status` is `spoofed`. */
  livenessScore?: number
}
