import { useEffect, useState } from 'react';
import { AppState, Platform } from 'react-native';
import { useCameraDevice, useCameraPermission, useFrameOutput } from 'react-native-vision-camera';
import { type EnrollFaceStatus, useFaceRecognizer } from 'react-native-vision-camera-face-recognizer';
import { createSynchronizable, scheduleOnRN } from 'react-native-worklets';

export type FaceEvent =
  | { status: 'none' | 'collecting' | 'unmatched' | 'spoofed' }
  | { status: 'matched'; employeeId: string; similarity: number };

/** Stable callback the camera thread can reach through scheduleOnRN; the listener can change freely. */
const channel = <T>() => {
  let listener: ((value: T) => void) | null = null;
  return {
    deliver: (value: T) => listener?.(value),
    listen: (next: ((value: T) => void) | null) => {
      listener = next;
    },
  };
};

// Tuned like Margelo's demo; liveness on so photos and screens never punch.
const RECOGNIZER_OPTIONS = {
  provider: 'auto',
  threads: 4,
  detection: { maxFaces: 1, threshold: 0.6, minFaceSize: 48, inputSize: 128 },
  recognition: {
    candidateRetrieval: 'centroid',
    threshold: 0.5,
    minimumMargin: 0.05,
    samplesPerDecision: 5,
    interval: 10,
    matchedInterval: 45,
    minFaceSize: 72,
  },
  tracking: { enabled: true, detectionInterval: 10 },
  liveness: { enabled: true, threshold: 0.5 },
} as const;

// Module-level on purpose: useFrameOutput compares these by reference, and a new object per render
// rebuilds the frame output and its worklet runtime, which crashes react-native-worklets 0.13 natively.
const TARGET_RESOLUTION = { width: 640, height: 480 };

export function useFaceEngine() {
  const device = useCameraDevice('front');
  const { hasPermission, canRequestPermission, requestPermission } = useCameraPermission();
  const [foreground, setForeground] = useState(AppState.currentState === 'active');

  useEffect(() => {
    if (canRequestPermission) requestPermission();
  }, [canRequestPermission, requestPermission]);

  useEffect(() => {
    const subscription = AppState.addEventListener('change', (next) => setForeground(next === 'active'));
    return () => subscription.remove();
  }, []);

  const { recognizer, error } = useFaceRecognizer({ isActive: foreground, ...RECOGNIZER_OPTIONS });

  const [bridge] = useState(() => ({
    // Employee id to capture an enrollment sample for on the next frame.
    enrollRequest: createSynchronizable<string | null>(null),
    lastFaceKey: createSynchronizable(''),
    faces: channel<FaceEvent>(),
    enrollments: channel<EnrollFaceStatus>(),
  }));

  const frameOutput = useFrameOutput({
    targetResolution: TARGET_RESOLUTION,
    pixelFormat: 'yuv',
    enablePhysicalBufferRotation: Platform.OS === 'ios',
    enablePreviewSizedOutputBuffers: true,
    allowDeferredStart: true,
    onFrame(frame) {
      'worklet';
      try {
        if (recognizer == null) return;
        const enrollId = bridge.enrollRequest.getDirty();
        if (enrollId != null) {
          bridge.enrollRequest.setBlocking(null);
          scheduleOnRN(bridge.enrollments.deliver, recognizer.enrollFace(enrollId, frame).status);
          return;
        }
        const face = recognizer.recognizeFaces(frame)[0];
        const event: FaceEvent =
          face == null
            ? { status: 'none' }
            : face.status === 'matched'
              ? { status: 'matched', employeeId: face.enrollmentId, similarity: face.similarity }
              : { status: face.status };
        // Cross to the JS thread when what the kiosk shows changes, and on every fresh match
        // (about every 1.5 s): someone who stays in view past the 15-minute block must still punch.
        const key = event.status === 'matched' ? `matched:${event.employeeId}` : event.status;
        const freshMatch = face != null && face.status === 'matched' && face.matchAge === 0;
        if (freshMatch || key !== bridge.lastFaceKey.getDirty()) {
          bridge.lastFaceKey.setBlocking(key);
          scheduleOnRN(bridge.faces.deliver, event);
        }
      } catch (e) {
        console.error('Face frame failed', e instanceof Error ? e.message : String(e));
      } finally {
        frame.dispose();
      }
    },
  });

  return { device, hasPermission, isActive: foreground, recognizer, error, frameOutput, bridge };
}
