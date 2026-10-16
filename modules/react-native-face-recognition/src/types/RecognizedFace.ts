import type { Point } from './Point'
import type { Rect } from './Rect'

/**
 * Recognition state for a detected face.
 *
 * Use this as the discriminator for `RecognizedFace`; status-specific fields
 * are only present on the matching union member.
 */
export type FaceRecognitionStatus =
  | 'collecting'
  | 'matched'
  | 'unmatched'
  | 'spoofed'

/**
 * Geometry and detector output shared by every recognized face variant.
 *
 * Coordinates are already transformed by native code into upright display
 * space, so overlays can map them directly onto the displayed camera preview.
 */
export interface BaseRecognizedFace {
  /** Bounding rectangle in oriented upright-display pixels. */
  bounds: Rect
  /** Detection confidence from `0` to `1`. */
  confidence: number
  /** Five facial landmarks in oriented upright-display pixels. */
  landmarks: Point[]
}

/** Face waiting for enough fresh embeddings to make one recognition decision. */
export interface CollectingFace extends BaseRecognizedFace {
  status: 'collecting'
  /** Fresh embeddings collected for the pending decision. */
  samplesCollected: number
  /** Fresh embeddings required for the pending decision. */
  samplesRequired: number
}

/** Face returned with the closest enrolled identity above the recognition threshold. */
export interface MatchedFace extends BaseRecognizedFace {
  status: 'matched'
  /** Application-provided identifier used during enrollment. */
  enrollmentId: string
  /** Cosine similarity from `-1` to `1`. */
  similarity: number
  /** Number of processed frames since this match was computed. */
  matchAge: number
}

/** Face returned without any enrolled identity above the recognition threshold. */
export interface UnmatchedFace extends BaseRecognizedFace {
  status: 'unmatched'
}

/** Face rejected by configured passive liveness checks. */
export interface SpoofedFace extends BaseRecognizedFace {
  status: 'spoofed'
  /** Passive liveness probability from `0` to `1`. */
  livenessScore: number
}

/**
 * Public face result returned from `FaceRecognizer.recognizeFaces`.
 *
 * Check `status` before reading match or liveness fields so TypeScript narrows
 * the result to the correct variant.
 */
export type RecognizedFace =
  | CollectingFace
  | MatchedFace
  | UnmatchedFace
  | SpoofedFace
