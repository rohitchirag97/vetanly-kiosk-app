export { FaceRecognition } from './FaceRecognition'
export type { FaceRecognizer } from './types/FaceRecognizer'
export type { FaceRecognizerFactory } from './specs/FaceRecognizerFactory.nitro'
export type {
  EnrollFaceResult,
  EnrollFaceStatus,
} from './types/EnrollFaceResult'
export type {
  FaceDetectionOptions,
  CandidateRetrieval,
  FaceExecutionProvider,
  FaceLivenessOptions,
  FaceRecognitionOptions,
  FaceRecognizerOptions,
  FaceTrackingOptions,
} from './types/FaceRecognizerOptions'
export type { Point } from './types/Point'
export type {
  BaseRecognizedFace,
  CollectingFace,
  FaceRecognitionStatus,
  MatchedFace,
  RecognizedFace,
  SpoofedFace,
  UnmatchedFace,
} from './types/RecognizedFace'
export type { Rect } from './types/Rect'
export {
  type UseFaceRecognizerOptions,
  type UseFaceRecognizerResult,
  useFaceRecognizer,
} from './hooks/useFaceRecognizer'
