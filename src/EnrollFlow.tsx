import { useEffect, useMemo, useState } from 'react';
import { ActivityIndicator, FlatList, Pressable, StyleSheet, Text, TextInput, View } from 'react-native';
import type { EnrollFaceStatus, FaceRecognizer } from 'react-native-vision-camera-face-recognizer';

import { decodeEnrollment } from './enrollment-buffer';
import { type Employee, type KioskCredentials, uploadTemplate } from './kiosk';
import { Button, colors, errorMessage, ui } from './ui';
import type { useFaceEngine } from './useFaceEngine';

type Bridge = ReturnType<typeof useFaceEngine>['bridge'];

const TARGET_SAMPLES = 5;
const MAX_MISSES = 25;
const CAPTURE_GAP_MS = 650;
const POSES = [
  'Look straight at the camera',
  'Turn your head slightly left',
  'Turn your head slightly right',
  'Lift your chin a little',
  'Lower your chin a little',
];
const RETRY_HINTS: Partial<Record<EnrollFaceStatus, string>> = {
  redundant: 'Hold a new angle, that one is already captured',
  noFace: 'Center your face in the frame',
  multipleFaces: 'Only one person in the frame, please',
  spoof: 'Use a real face, not a photo or screen',
  lowQuality: 'Face the camera in good light',
};

type Props = {
  kiosk: KioskCredentials;
  recognizer: FaceRecognizer;
  bridge: Bridge;
  employees: Employee[];
  enrolledIds: Set<string>;
  /** Called on every exit; the caller reloads faces since capture replaces the employee's samples. */
  onClose: () => void;
};

export function EnrollFlow({ kiosk, recognizer, bridge, employees, enrolledIds, onClose }: Props) {
  const [step, setStep] = useState<'pick' | 'consent' | 'capture'>('pick');
  const [employee, setEmployee] = useState<Employee | null>(null);
  const [attempt, setAttempt] = useState(0);

  if (step === 'pick' || !employee) {
    return (
      <PickStep
        employees={employees}
        enrolledIds={enrolledIds}
        onPick={(picked) => {
          setEmployee(picked);
          setStep('consent');
        }}
        onClose={onClose}
      />
    );
  }
  if (step === 'consent') {
    return <ConsentStep employee={employee} onAgree={() => setStep('capture')} onBack={() => setStep('pick')} />;
  }
  return (
    <CaptureStep
      key={attempt}
      kiosk={kiosk}
      recognizer={recognizer}
      bridge={bridge}
      employee={employee}
      onRetry={() => setAttempt((n) => n + 1)}
      onNext={() => setStep('pick')}
      onClose={onClose}
    />
  );
}

function PickStep({
  employees,
  enrolledIds,
  onPick,
  onClose,
}: {
  employees: Employee[];
  enrolledIds: Set<string>;
  onPick: (employee: Employee) => void;
  onClose: () => void;
}) {
  const [query, setQuery] = useState('');
  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    return q
      ? employees.filter((e) => e.name.toLowerCase().includes(q) || e.employeeCode?.toLowerCase().includes(q))
      : employees;
  }, [employees, query]);

  return (
    <View style={[ui.panel, ui.tall]}>
      <Text style={ui.title}>Who is enrolling?</Text>
      <TextInput
        accessibilityLabel="Search employees"
        placeholder="Search by name or employee code"
        value={query}
        onChangeText={setQuery}
        style={styles.search}
      />
      <FlatList
        style={styles.list}
        data={matches}
        keyExtractor={(e) => e.id}
        keyboardShouldPersistTaps="handled"
        ListEmptyComponent={<Text style={ui.hint}>No employees found.</Text>}
        renderItem={({ item }) => (
          <Pressable accessibilityRole="button" onPress={() => onPick(item)} style={styles.item}>
            <Text style={styles.itemName}>{item.name}</Text>
            <Text style={styles.itemMeta}>
              {[item.employeeCode, enrolledIds.has(item.id) ? 'Enrolled, will be replaced' : 'Not enrolled']
                .filter(Boolean)
                .join(' · ')}
            </Text>
          </Pressable>
        )}
      />
      <Button label="Done" onPress={onClose} secondary />
    </View>
  );
}

// Biometric consent (Google Play policy, India's DPDP Act): the employee agrees before any face data is captured.
function ConsentStep({ employee, onAgree, onBack }: { employee: Employee; onAgree: () => void; onBack: () => void }) {
  return (
    <View style={ui.panel}>
      <Text style={ui.title}>Face data consent</Text>
      <Text style={styles.consentText}>
        {employee.name}, to mark your attendance with your face, this kiosk will scan your face and save a face
        template: a set of numbers made from your face, not a photo.
      </Text>
      <Text style={styles.consentText}>
        It is stored securely by Vetanly for your employer and used only to recognise you when you mark attendance.
        Your employer can delete it at any time; ask your manager if you want it removed.
      </Text>
      <Text style={styles.consentText}>Do you agree?</Text>
      <View style={ui.row}>
        <Button label="No, go back" onPress={onBack} secondary />
        <Button label="I agree" onPress={onAgree} />
      </View>
    </View>
  );
}

function CaptureStep({
  kiosk,
  recognizer,
  bridge,
  employee,
  onRetry,
  onNext,
  onClose,
}: {
  kiosk: KioskCredentials;
  recognizer: FaceRecognizer;
  bridge: Bridge;
  employee: Employee;
  onRetry: () => void;
  onNext: () => void;
  onClose: () => void;
}) {
  const [captured, setCaptured] = useState(0);
  const [hint, setHint] = useState(POSES[0]);
  const [phase, setPhase] = useState<'capturing' | 'saving' | 'done' | 'error'>('capturing');

  useEffect(() => {
    let cancelled = false;
    let count = 0;
    let misses = 0;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const fail = (text: string) => {
      setPhase('error');
      setHint(text);
    };
    const requestSample = () => {
      timer = setTimeout(() => bridge.enrollRequest.setBlocking(employee.id), CAPTURE_GAP_MS);
    };

    const save = async () => {
      setPhase('saving');
      try {
        for (const embedding of decodeEnrollment(recognizer.getEnrollment(employee.id))) {
          await uploadTemplate(kiosk, employee.id, embedding);
        }
        if (!cancelled) setPhase('done');
      } catch (e) {
        if (!cancelled) fail(errorMessage(e));
      }
    };

    // Start from an empty sample set so old samples do not mix with this capture.
    recognizer.removeEnrollment(employee.id);
    bridge.enrollments.listen((status) => {
      if (cancelled) return;
      if (status === 'enrolled') {
        count++;
        setCaptured(count);
        if (count >= TARGET_SAMPLES) {
          save();
          return;
        }
        setHint(POSES[count]);
        return requestSample();
      }
      if (++misses >= MAX_MISSES) return fail(`${RETRY_HINTS[status] ?? 'Capture failed'}. Try again.`);
      setHint(RETRY_HINTS[status] ?? 'Try again');
      requestSample();
    });
    requestSample();

    return () => {
      cancelled = true;
      clearTimeout(timer);
      bridge.enrollments.listen(null);
      bridge.enrollRequest.setBlocking(null);
    };
  }, [kiosk, recognizer, bridge, employee]);

  return (
    <>
      <View style={[styles.faceGuide, phase === 'error' && styles.faceGuideError]} pointerEvents="none" />
      <View style={styles.captureCard}>
        <Text style={styles.captureEyebrow}>ENROLLING</Text>
        <Text style={styles.captureName} numberOfLines={1}>
          {employee.name}
        </Text>
        {phase === 'capturing' && (
          <>
            <Text style={styles.captureHint}>{hint}</Text>
            <View style={styles.dots} accessibilityLabel={`${captured} of ${TARGET_SAMPLES} captured`}>
              {Array.from({ length: TARGET_SAMPLES }, (_, i) => (
                <View key={i} style={[styles.dot, i < captured && styles.dotDone]} />
              ))}
            </View>
            <Button label="Cancel" onPress={onClose} secondary />
          </>
        )}
        {phase === 'saving' && (
          <>
            <ActivityIndicator color={colors.brand} size="large" />
            <Text style={ui.hint}>Saving…</Text>
          </>
        )}
        {phase === 'done' && (
          <>
            <Text style={[styles.captureHint, styles.success]}>Enrolled. They can now mark attendance.</Text>
            <View style={ui.row}>
              <Button label="Finish" onPress={onClose} secondary />
              <Button label="Enroll another" onPress={onNext} />
            </View>
          </>
        )}
        {phase === 'error' && (
          <>
            <Text style={[styles.captureHint, styles.failure]}>{hint}</Text>
            <View style={ui.row}>
              <Button label="Close" onPress={onClose} secondary />
              <Button label="Try again" onPress={onRetry} />
            </View>
          </>
        )}
      </View>
    </>
  );
}

const styles = StyleSheet.create({
  search: {
    fontSize: 17,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 14,
    paddingVertical: 12,
    paddingHorizontal: 16,
    color: colors.text,
    backgroundColor: '#F9FAFB',
  },
  list: { flex: 1 },
  consentText: { fontSize: 16, lineHeight: 23, color: colors.text },
  item: { paddingVertical: 14, borderBottomWidth: StyleSheet.hairlineWidth, borderColor: colors.border },
  itemName: { fontSize: 17, fontWeight: '600', color: colors.text },
  itemMeta: { fontSize: 14, color: colors.muted, marginTop: 2 },
  faceGuide: {
    position: 'absolute',
    alignSelf: 'center',
    top: '18%',
    width: '62%',
    aspectRatio: 0.78,
    borderRadius: 999,
    borderWidth: 4,
    borderColor: colors.brand,
  },
  faceGuideError: { borderColor: colors.danger },
  captureCard: {
    position: 'absolute',
    left: 20,
    right: 20,
    bottom: 36,
    alignItems: 'center',
    gap: 12,
    padding: 22,
    borderRadius: 24,
    backgroundColor: '#fff',
    elevation: 8,
  },
  captureEyebrow: { color: colors.brand, fontSize: 12, fontWeight: '700', letterSpacing: 2 },
  captureName: { color: colors.text, fontSize: 22, fontWeight: '700' },
  captureHint: { color: colors.text, fontSize: 19, fontWeight: '500', textAlign: 'center' },
  success: { color: colors.success },
  failure: { color: colors.danger },
  dots: { flexDirection: 'row', gap: 10 },
  dot: { width: 14, height: 14, borderRadius: 7, backgroundColor: colors.border },
  dotDone: { backgroundColor: colors.brand },
});
