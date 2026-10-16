/**
 * ONNX Runtime execution provider requested by the native engine.
 *
 * - `auto` requests NNAPI on Android with XNNPACK CPU fallback.
 * - `gpu` requests GPU execution; Android falls back to NNAPI/XNNPACK when QNN is unavailable.
 * - `cpu` forces XNNPACK CPU execution.
 */
export type FaceExecutionProvider = 'auto' | 'gpu' | 'cpu'

/** Coarse identity retrieval policy before per-template verification. */
export type CandidateRetrieval = 'centroid' | 'templateMax'

/**
 * Runtime options consumed by the bundled native face-recognition pipeline.
 *
 * These values are captured when a recognizer is created. Create a new
 * recognizer to apply option changes; frame processors should not mutate this
 * object while recognition is running.
 */
export interface FaceRecognizerOptions {
  /** Requested ONNX Runtime provider for detector, embedding, and liveness models. */
  provider: FaceExecutionProvider
  /** Intra-op threads for embedding and liveness; detection uses at most two. */
  threads: number
  /** Face detector policy. */
  detection: FaceDetectionOptions
  /** Face matching policy. */
  recognition: FaceRecognitionOptions
  /** Native tracking policy. */
  tracking: FaceTrackingOptions
  /** Anti-spoofing policy. */
  liveness: FaceLivenessOptions
}

/**
 * Controls native face detection cadence and candidate filtering.
 *
 * Detection is the most frequent expensive stage. Prefer the smallest input
 * size and face count that still work for the camera distance in your UI.
 */
export interface FaceDetectionOptions {
  /** Maximum faces returned per frame. Lower values reduce sorting and matching work. */
  maxFaces: number
  /** Minimum detector confidence from `0` to `1`. */
  threshold: number
  /** Minimum face width and height in source-frame pixels. */
  minFaceSize: number
  /** Square detector input size in pixels. Smaller is faster. */
  inputSize: number
}

/**
 * Controls lightweight native tracking between full detector passes.
 *
 * Tracking keeps overlay positions stable and avoids running detection on every
 * frame when the camera scene is mostly continuous.
 */
export interface FaceTrackingOptions {
  /** Enables native face tracking between detector refreshes. */
  enabled: boolean
  /** Full detection refresh interval, measured in processed frames. */
  detectionInterval: number
}

/**
 * Controls embedding extraction and identity matching.
 *
 * Recognition is more expensive than detection, so the interval knobs let the
 * engine reuse recent matches while keeping the UI responsive.
 */
export interface FaceRecognitionOptions {
  /**
   * Chooses how identities enter template verification. `templateMax` retains
   * an identity when any enrolled template is close to any decision frame.
   */
  candidateRetrieval: CandidateRetrieval
  /** Minimum cosine similarity required for a match. */
  threshold: number
  /** Minimum score difference between the best and runner-up identities. */
  minimumMargin: number
  /** Consecutive fresh face embeddings combined for each decision. Integer from `1` to `5`. */
  samplesPerDecision: number
  /** Embedding/matching interval, measured in processed frames. */
  interval: number
  /** Embedding/matching interval after a stable match, measured in processed frames. */
  matchedInterval: number
  /** Minimum face width/height required before recognition is attempted. */
  minFaceSize: number
}

/**
 * Controls passive liveness checks used to reject likely spoof attempts.
 *
 * Enable this when enrollment or recognition must reject photos/screens, and
 * keep it disabled when latency is more important than spoof resistance.
 */
export interface FaceLivenessOptions {
  /** Enables passive anti-spoofing checks for enrollment and matching. */
  enabled: boolean
  /** Minimum live probability from `0` to `1`. */
  threshold: number
}
