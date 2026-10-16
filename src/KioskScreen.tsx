import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Image, Pressable, StyleSheet, Text, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Camera } from 'react-native-vision-camera';

import {
  applySettings,
  DEFAULT_SETTINGS,
  type DeviceStatus,
  KioskDevice,
  type KioskSettings,
  loadSettings,
  punchFeedback,
  saveSettings,
} from './device-settings';
import { EnrollFlow } from './EnrollFlow';
import { encodeEnrollment } from './enrollment-buffer';
import {
  type Directory,
  type KioskCredentials,
  loadDirectory,
  punch,
  syncDirectory,
  UnauthorizedError,
} from './kiosk';
import { ManagerCode } from './ManagerCode';
import { NetworkBars, SettingsPanel } from './SettingsPanel';
import { colors } from './ui';
import { type FaceEvent, useFaceEngine } from './useFaceEngine';

// Matches the server rule: the same face within 15 minutes is the same visit.
const COOLDOWN_MS = 15 * 60_000;
// After a refusal (e.g. on leave), wait a little before asking the server again.
const REJECTED_RETRY_MS = 60_000;
const BANNER_MS = 4_000;
const SYNC_MS = 5 * 60_000;
const DEVICE_STATUS_MS = 10_000;
// One manager code opens Enroll and Settings for a while; kept under the server's 15-minute enrollment window.
const MANAGER_UNLOCK_MS = 14 * 60_000;

const LOGO = require('./assets/vetanly-icon.png');

type Mode = 'scan' | 'enroll' | 'settings';

type Banner = { name: string; label: string; detail: string; tone: 'in' | 'out' | 'warn' | 'error' };

const TONE_COLOR: Record<Banner['tone'], string> = {
  in: colors.success,
  out: colors.brand,
  warn: colors.warning,
  error: colors.danger,
};

const PROMPTS: Record<FaceEvent['status'], { title: string; hint: string }> = {
  none: { title: 'Look at the camera', hint: 'Stand in front of the kiosk to mark your attendance' },
  collecting: { title: 'Hold still…', hint: 'Recognizing your face' },
  matched: { title: 'Recognized', hint: 'Marking your attendance' },
  unmatched: { title: 'Face not recognized', hint: 'Ask a manager to enroll you' },
  spoofed: { title: 'Use your real face', hint: 'Photos and screens are not accepted' },
};

const RING_COLOR: Record<FaceEvent['status'], string> = {
  none: 'rgba(255,255,255,0.85)',
  collecting: colors.warning,
  matched: colors.success,
  unmatched: colors.danger,
  spoofed: colors.danger,
};

const timeOf = (iso: string) => new Date(iso).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });

const initials = (name: string) =>
  name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0].toUpperCase())
    .join('');

type Props = { kiosk: KioskCredentials; online: boolean; onUnauthorized: () => void };

export function KioskScreen({ kiosk, online, onUnauthorized }: Props) {
  const insets = useSafeAreaInsets();
  const engine = useFaceEngine();
  const { recognizer, bridge } = engine;
  const [directory, setDirectory] = useState<Directory>({ employees: [], templates: [] });
  const [mode, setMode] = useState<Mode>('scan');
  const [codeFor, setCodeFor] = useState<Exclude<Mode, 'scan'> | null>(null);
  const unlockedUntil = useRef(0);
  const enrolling = mode === 'enroll';
  const scanning = mode === 'scan' && codeFor === null;
  const [settings, setSettings] = useState<KioskSettings>(DEFAULT_SETTINGS);
  const [device, setDevice] = useState<DeviceStatus | null>(null);
  const [face, setFace] = useState<FaceEvent['status']>('none');
  const [banner, setBanner] = useState<Banner | null>(null);
  const recent = useRef(new Map<string, { until: number; banner: Banner }>());
  const inFlight = useRef(new Set<string>());
  const bannerTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const names = useMemo(() => new Map(directory.employees.map((e) => [e.id, e.name])), [directory]);
  const enrolledIds = useMemo(() => new Set(directory.templates.map((t) => t.employeeId)), [directory]);

  const showBanner = useCallback((next: Banner) => {
    setBanner(next);
    clearTimeout(bannerTimer.current);
    bannerTimer.current = setTimeout(() => setBanner(null), BANNER_MS);
  }, []);

  const sync = useCallback(async () => {
    try {
      setDirectory(await syncDirectory(kiosk));
    } catch (e) {
      if (e instanceof UnauthorizedError) onUnauthorized();
      // Offline: keep recognizing with the faces already on the device.
    }
  }, [kiosk, onUnauthorized]);

  useEffect(() => {
    loadDirectory().then(setDirectory);
    sync();
  }, [sync]);

  useEffect(() => {
    // Enrollment edits the recognizer's samples; a reload mid-capture would wipe them.
    if (enrolling) return;
    const timer = setInterval(sync, SYNC_MS);
    return () => clearInterval(timer);
  }, [enrolling, sync]);

  // Load every employee's samples into the native matcher whenever the synced set changes.
  useEffect(() => {
    if (!recognizer || enrolling) return;
    recognizer.clearEnrollments();
    const byEmployee = new Map<string, number[][]>();
    for (const t of directory.templates) {
      byEmployee.set(t.employeeId, [...(byEmployee.get(t.employeeId) ?? []), t.embedding]);
    }
    for (const [employeeId, samples] of byEmployee) {
      try {
        // The native set holds at most 8 samples of one length; the server keeps 5.
        const usable = samples.filter((s) => s.length === samples[0].length).slice(-8);
        recognizer.addEnrollment(employeeId, encodeEnrollment(usable));
      } catch (e) {
        console.warn('Skipping unusable face template', employeeId, e);
      }
    }
  }, [recognizer, directory, enrolling]);

  const handleMatch = useCallback(
    async (employeeId: string, similarity: number) => {
      const name = names.get(employeeId) ?? 'Employee';
      const previous = recent.current.get(employeeId);
      if (previous && previous.until > Date.now()) {
        return showBanner({
          ...previous.banner,
          detail: `${previous.banner.detail} · next punch after ${timeOf(new Date(previous.until).toISOString())}`,
        });
      }
      if (inFlight.current.has(employeeId)) return;
      inFlight.current.add(employeeId);
      try {
        const result = await punch(kiosk, employeeId, similarity);
        const checkIn = result?.eventType === 'check_in';
        const next: Banner =
          result === null
            ? { name, label: 'Saved offline', detail: 'Will sync when the kiosk is back online', tone: 'warn' }
            : {
                name,
                label: `${result.alreadyRecorded ? 'Already checked' : 'Checked'} ${checkIn ? 'in' : 'out'}`,
                detail: `${checkIn ? 'Have a great day' : 'See you next time'} · ${timeOf(result.occurredAt)}`,
                tone: checkIn ? 'in' : 'out',
              };
        recent.current.set(employeeId, { until: Date.now() + COOLDOWN_MS, banner: next });
        showBanner(next);
        // Only for a fresh result: the repeats shown during the cooldown stay silent.
        punchFeedback(settings, true);
      } catch (e) {
        if (e instanceof UnauthorizedError) return onUnauthorized();
        const next: Banner = {
          name,
          label: "Couldn't mark attendance",
          detail: e instanceof Error ? e.message : String(e),
          tone: 'error',
        };
        recent.current.set(employeeId, { until: Date.now() + REJECTED_RETRY_MS, banner: next });
        showBanner(next);
        punchFeedback(settings, false);
      } finally {
        inFlight.current.delete(employeeId);
      }
    },
    [kiosk, names, onUnauthorized, showBanner, settings],
  );

  const refreshDevice = useCallback(() => {
    KioskDevice?.getStatus().then(setDevice).catch(() => {});
  }, []);

  useEffect(() => {
    loadSettings().then((loaded) => {
      setSettings(loaded);
      applySettings(loaded);
    });
    refreshDevice();
    const timer = setInterval(refreshDevice, DEVICE_STATUS_MS);
    return () => clearInterval(timer);
  }, [refreshDevice]);

  const changeSettings = (next: KioskSettings) => {
    setSettings(next);
    applySettings(next);
    saveSettings(next);
    // Pinning and lock flags settle asynchronously; read the phone again shortly after.
    setTimeout(refreshDevice, 800);
  };

  const openManagerScreen = (target: Exclude<Mode, 'scan'>) => {
    if (Date.now() < unlockedUntil.current) setMode(target);
    else setCodeFor(target);
  };

  // Faces are ignored while a manager screen is open, so nobody punches by standing nearby.
  useEffect(() => {
    if (!scanning) return;
    bridge.faces.listen((event) => {
      setFace(event.status);
      if (event.status === 'matched') handleMatch(event.employeeId, event.similarity);
    });
    return () => bridge.faces.listen(null);
  }, [bridge, scanning, handleMatch]);

  useEffect(() => () => clearTimeout(bannerTimer.current), []);

  if (engine.error) return <Message text={`Face recognition failed to start: ${engine.error.message}`} />;
  if (!engine.hasPermission) return <Message text="Allow camera access to mark attendance." />;
  if (!engine.device) return <Message text="No front camera found on this device." />;

  const prompt =
    !recognizer
      ? { title: 'Starting…', hint: 'Loading face recognition' }
      : enrolledIds.size === 0
        ? { title: 'No faces enrolled yet', hint: 'A manager can tap Enroll to add employees' }
        : PROMPTS[face];
  const timeWarning = settings.requireAutoTime && device && !(device.autoTime && device.autoTimeZone);

  return (
    <View style={styles.root}>
      <Camera
        style={StyleSheet.absoluteFill}
        device={engine.device}
        isActive={engine.isActive}
        mirrorMode="on"
        orientationSource="interface"
        // No resolutionBias on the 640x480 frame output: it would drag the whole session (and the preview) down to VGA.
        constraints={[{ fps: 30 }]}
        outputs={[engine.frameOutput]}
      />

      <View style={[styles.header, { paddingTop: insets.top + 12 }]}>
        <View style={styles.headerRow}>
          <Image source={LOGO} style={styles.logo} accessibilityIgnoresInvertColors />
          <View style={styles.brandBlock}>
            <Text style={styles.brand}>VETANLY</Text>
            <Text style={styles.kioskName} numberOfLines={1}>
              {kiosk.name}
            </Text>
          </View>
          <View style={styles.statusPill}>
            {settings.showNetwork && device && <NetworkBars status={device} />}
            <View style={[styles.dot, { backgroundColor: online ? colors.success : colors.danger }]} />
            <Text style={styles.statusText}>{online ? 'Online' : 'Offline'}</Text>
          </View>
        </View>
        {scanning && <Clock />}
        {scanning && timeWarning && (
          <View style={styles.warning} accessibilityRole="alert">
            <Text style={styles.warningText}>
              Phone date & time isn't automatic. Attendance times may be wrong.
            </Text>
          </View>
        )}
      </View>

      {scanning && <View style={[styles.faceGuide, { borderColor: RING_COLOR[recognizer ? face : 'none'] }]} pointerEvents="none" />}

      {codeFor ? (
        <ManagerCode
          kiosk={kiosk}
          onUnlocked={() => {
            unlockedUntil.current = Date.now() + MANAGER_UNLOCK_MS;
            setMode(codeFor);
            setCodeFor(null);
          }}
          onClose={() => setCodeFor(null)}
        />
      ) : mode === 'settings' ? (
        <SettingsPanel settings={settings} status={device} onChange={changeSettings} onClose={() => setMode('scan')} />
      ) : enrolling && recognizer ? (
        <EnrollFlow
          kiosk={kiosk}
          recognizer={recognizer}
          bridge={bridge}
          employees={directory.employees}
          enrolledIds={enrolledIds}
          onClose={() => {
            setMode('scan');
            sync();
          }}
        />
      ) : (
        <View style={[styles.footer, { paddingBottom: insets.bottom + 20 }]}>
          {banner ? (
            <View style={styles.resultCard} accessibilityLiveRegion="polite">
              <View style={[styles.avatar, { backgroundColor: TONE_COLOR[banner.tone] }]}>
                <Text style={styles.avatarText}>{initials(banner.name)}</Text>
              </View>
              <View style={styles.resultBody}>
                <Text style={styles.resultName} numberOfLines={1}>
                  {banner.name}
                </Text>
                <Text style={[styles.resultLabel, { color: TONE_COLOR[banner.tone] }]}>{banner.label}</Text>
                <Text style={styles.resultDetail} numberOfLines={2}>
                  {banner.detail}
                </Text>
              </View>
            </View>
          ) : (
            <View style={styles.promptCard} accessibilityLiveRegion="polite">
              <Text style={styles.promptTitle}>{prompt.title}</Text>
              <Text style={styles.promptHint}>{prompt.hint}</Text>
            </View>
          )}
          <View style={styles.managerRow}>
            <ManagerLink label="Settings" onPress={() => openManagerScreen('settings')} />
            {recognizer && <ManagerLink label="Enroll" onPress={() => openManagerScreen('enroll')} />}
          </View>
        </View>
      )}
    </View>
  );
}

/** Large wall clock; refreshes often enough that the minute never looks stale. */
function Clock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const timer = setInterval(() => setNow(new Date()), 5_000);
    return () => clearInterval(timer);
  }, []);
  return (
    <View style={styles.clock}>
      <Text style={styles.clockTime}>{now.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}</Text>
      <Text style={styles.clockDate}>
        {now.toLocaleDateString([], { weekday: 'long', day: 'numeric', month: 'long' })}
      </Text>
    </View>
  );
}

/** Manager entry points stay small so employees focus on the camera. */
function ManagerLink({ label, onPress }: { label: string; onPress: () => void }) {
  return (
    <Pressable accessibilityRole="button" onPress={onPress} hitSlop={12} style={styles.managerLink}>
      <Text style={styles.managerLinkText}>{label}</Text>
    </Pressable>
  );
}

function Message({ text }: { text: string }) {
  return (
    <View style={styles.message}>
      <Text style={styles.messageText}>{text}</Text>
    </View>
  );
}

// Keeps white text readable over any camera background without hard-edged overlays.
const shadow = { textShadowColor: 'rgba(0,0,0,0.6)', textShadowOffset: { width: 0, height: 1 }, textShadowRadius: 8 };

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#000' },
  header: { position: 'absolute', top: 0, left: 0, right: 0, paddingHorizontal: 20, gap: 12 },
  headerRow: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  logo: { width: 40, height: 40 },
  brandBlock: { flex: 1 },
  brand: { color: 'rgba(255,255,255,0.85)', fontSize: 12, fontWeight: '700', letterSpacing: 2, ...shadow },
  kioskName: { color: '#fff', fontSize: 20, fontWeight: '700', ...shadow },
  statusPill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 999,
    backgroundColor: 'rgba(0,0,0,0.45)',
  },
  dot: { width: 8, height: 8, borderRadius: 4 },
  statusText: { color: '#fff', fontSize: 14, fontWeight: '600' },
  clock: { alignItems: 'center', marginTop: 4 },
  clockTime: { color: '#fff', fontSize: 56, fontWeight: '300', letterSpacing: 1, ...shadow },
  clockDate: { color: '#fff', fontSize: 17, fontWeight: '500', ...shadow },
  warning: { alignSelf: 'center', paddingVertical: 8, paddingHorizontal: 14, borderRadius: 12, backgroundColor: colors.warning },
  warningText: { color: '#fff', fontSize: 14, fontWeight: '600', textAlign: 'center' },
  faceGuide: {
    position: 'absolute',
    alignSelf: 'center',
    top: '30%',
    width: '62%',
    aspectRatio: 0.78,
    borderRadius: 999,
    borderWidth: 4,
  },
  footer: { position: 'absolute', left: 0, right: 0, bottom: 0, paddingHorizontal: 20, gap: 14 },
  promptCard: {
    alignItems: 'center',
    gap: 4,
    paddingVertical: 18,
    paddingHorizontal: 20,
    borderRadius: 24,
    backgroundColor: 'rgba(17,17,27,0.72)',
  },
  promptTitle: { color: '#fff', fontSize: 24, fontWeight: '700', textAlign: 'center' },
  promptHint: { color: 'rgba(255,255,255,0.8)', fontSize: 16, textAlign: 'center' },
  resultCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
    padding: 18,
    borderRadius: 24,
    backgroundColor: '#fff',
    elevation: 8,
  },
  avatar: { width: 64, height: 64, borderRadius: 32, alignItems: 'center', justifyContent: 'center' },
  avatarText: { color: '#fff', fontSize: 24, fontWeight: '700' },
  resultBody: { flex: 1, gap: 2 },
  resultName: { color: colors.text, fontSize: 22, fontWeight: '700' },
  resultLabel: { fontSize: 18, fontWeight: '700' },
  resultDetail: { color: colors.muted, fontSize: 15 },
  managerRow: { flexDirection: 'row', justifyContent: 'center', gap: 24 },
  managerLink: { paddingVertical: 4, paddingHorizontal: 8 },
  managerLinkText: { color: 'rgba(255,255,255,0.85)', fontSize: 14, fontWeight: '600', ...shadow },
  message: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 32, backgroundColor: '#fff' },
  messageText: { fontSize: 20, textAlign: 'center', color: '#111' },
});
