import { useMemo, useSyncExternalStore } from 'react'
import { FaceRecognition } from '../FaceRecognition'
import type { FaceRecognizer } from '../types/FaceRecognizer'
import type { FaceRecognizerOptions } from '../types/FaceRecognizerOptions'

export interface UseFaceRecognizerOptions extends FaceRecognizerOptions {
  /**
   * Controls the native recognizer lifecycle.
   *
   * When `false`, the hook releases the current recognizer and clears the
   * result. Native resources are disposed after any in-flight worklet releases
   * its reference. Set this from camera/app foreground state so model memory is
   * not held while frames are not being processed.
   */
  isActive: boolean
}

export interface UseFaceRecognizerResult {
  /**
   * Ready native recognizer, or `undefined` while inactive, loading, or after an
   * initialization failure.
   */
  recognizer: FaceRecognizer | undefined
  /** Initialization error from the latest creation attempt. */
  error: Error | undefined
  /** `true` while native models are being loaded for the active options. */
  isLoading: boolean
}

function toError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error))
}

const INACTIVE: UseFaceRecognizerResult = {
  recognizer: undefined,
  error: undefined,
  isLoading: false,
}

const LOADING: UseFaceRecognizerResult = {
  recognizer: undefined,
  error: undefined,
  isLoading: true,
}

class FaceRecognizerStore {
  private readonly options: FaceRecognizerOptions | undefined
  private readonly listeners = new Set<() => void>()
  private snapshot: UseFaceRecognizerResult
  private runId = 0

  constructor(options: FaceRecognizerOptions | undefined) {
    this.options = options
    this.snapshot = options === undefined ? INACTIVE : LOADING
  }

  readonly subscribe = (onStoreChange: () => void): (() => void) => {
    this.listeners.add(onStoreChange)
    if (this.listeners.size === 1) {
      this.start()
    }

    return () => {
      this.listeners.delete(onStoreChange)
      if (this.listeners.size === 0) {
        this.stop()
      }
    }
  }

  readonly getSnapshot = (): UseFaceRecognizerResult => this.snapshot

  readonly getServerSnapshot = (): UseFaceRecognizerResult => INACTIVE

  private start(): void {
    const options = this.options
    if (options === undefined) {
      return
    }

    const runId = ++this.runId

    FaceRecognition.create(options)
      .then((recognizer: FaceRecognizer) => {
        if (runId !== this.runId) {
          recognizer.dispose()
          return
        }
        this.publish({ recognizer, error: undefined, isLoading: false })
      })
      .catch((creationError: unknown) => {
        if (runId !== this.runId) {
          return
        }
        this.publish({
          recognizer: undefined,
          error: toError(creationError),
          isLoading: false,
        })
      })
  }

  private stop(): void {
    this.runId++
    this.snapshot = this.options === undefined ? INACTIVE : LOADING
  }

  private publish(next: UseFaceRecognizerResult): void {
    this.snapshot = next
    for (const listener of this.listeners) {
      listener()
    }
  }
}

export function useFaceRecognizer(
  options: UseFaceRecognizerOptions
): UseFaceRecognizerResult {
  const { isActive, provider, threads } = options
  const {
    inputSize,
    maxFaces,
    minFaceSize: detectionMinFaceSize,
    threshold: detectionThreshold,
  } = options.detection
  const {
    candidateRetrieval,
    interval,
    matchedInterval,
    minimumMargin,
    minFaceSize: recognitionMinFaceSize,
    samplesPerDecision,
    threshold: recognitionThreshold,
  } = options.recognition
  const { enabled: trackingEnabled, detectionInterval } = options.tracking
  const { enabled: livenessEnabled, threshold: livenessThreshold } =
    options.liveness

  // Not a perf optimization: store identity *is* the native recognizer's
  // lifetime, and `bob` ships this package without React Compiler, so
  // memoization has to be explicit here.
  // react-doctor-disable-next-line react-doctor/react-compiler-no-manual-memoization
  const store = useMemo(
    () =>
      new FaceRecognizerStore(
        isActive
          ? {
              provider,
              threads,
              detection: {
                inputSize,
                maxFaces,
                minFaceSize: detectionMinFaceSize,
                threshold: detectionThreshold,
              },
              recognition: {
                candidateRetrieval,
                interval,
                matchedInterval,
                minimumMargin,
                minFaceSize: recognitionMinFaceSize,
                samplesPerDecision,
                threshold: recognitionThreshold,
              },
              tracking: { enabled: trackingEnabled, detectionInterval },
              liveness: {
                enabled: livenessEnabled,
                threshold: livenessThreshold,
              },
            }
          : undefined
      ),
    [
      isActive,
      provider,
      threads,
      inputSize,
      maxFaces,
      detectionMinFaceSize,
      detectionThreshold,
      candidateRetrieval,
      interval,
      matchedInterval,
      minimumMargin,
      recognitionMinFaceSize,
      samplesPerDecision,
      recognitionThreshold,
      trackingEnabled,
      detectionInterval,
      livenessEnabled,
      livenessThreshold,
    ]
  )

  return useSyncExternalStore(
    store.subscribe,
    store.getSnapshot,
    store.getServerSnapshot
  )
}
