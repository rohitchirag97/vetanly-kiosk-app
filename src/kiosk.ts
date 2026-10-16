import AsyncStorage from '@react-native-async-storage/async-storage';
import * as Keychain from 'react-native-keychain';

import { API_URL } from './config';

export type KioskCredentials = { kioskId: string; name: string; apiKey: string };
export type Employee = { id: string; name: string; employeeCode: string | null };
export type FaceTemplate = { id: string; employeeId: string; embedding: number[]; modelName: string };
export type PunchResult = {
  eventType: 'check_in' | 'check_out';
  occurredAt: string;
  alreadyRecorded: boolean;
};
type Punch = {
  employeeId: string;
  occurredAt: string;
  utcOffsetMinutes: number;
  externalEventId: string;
  faceMatchScore: number;
  livenessPassed: boolean;
};

export class UnauthorizedError extends Error {}
/** The server answered and refused; retrying the same request will not help. */
export class RejectedError extends Error {}

export const FACE_MODEL = { modelName: 'sface', modelVersion: '2021dec' };

const SERVICE = 'vetanly-kiosk';
// shortcut: kiosks paired before the rebrand stored their key here; drop once every kiosk has opened the new build.
const LEGACY_SERVICE = 'hazripro-kiosk';
const DIRECTORY_KEY = 'kiosk-directory';
const QUEUE_KEY = 'kiosk-punch-queue';
const REQUEST_TIMEOUT_MS = 8_000;

const saveCredentials = (kiosk: KioskCredentials) =>
  // Kiosks reboot unattended, so the key must be readable without a user unlocking the device again.
  Keychain.setGenericPassword(kiosk.kioskId, JSON.stringify(kiosk), {
    service: SERVICE,
    accessible: Keychain.ACCESSIBLE.AFTER_FIRST_UNLOCK_THIS_DEVICE_ONLY,
  });

export const loadCredentials = async (): Promise<KioskCredentials | null> => {
  const stored = await Keychain.getGenericPassword({ service: SERVICE });
  if (stored) return JSON.parse(stored.password);
  // Move a pre-rebrand key over once, so the kiosk stays paired.
  const legacy = await Keychain.getGenericPassword({ service: LEGACY_SERVICE });
  if (!legacy) return null;
  const kiosk: KioskCredentials = JSON.parse(legacy.password);
  await saveCredentials(kiosk);
  await Keychain.resetGenericPassword({ service: LEGACY_SERVICE });
  return kiosk;
};

export const clearCredentials = async () => {
  await Keychain.resetGenericPassword({ service: SERVICE });
  await Keychain.resetGenericPassword({ service: LEGACY_SERVICE });
  // Another organization may pair this device next, so drop its faces. Unsent punches are kept:
  // if the same kiosk is paired again they still sync, otherwise flushPunches discards them.
  await AsyncStorage.removeItem(DIRECTORY_KEY);
};

const request = async <T>(
  method: 'GET' | 'POST',
  path: string,
  body?: unknown,
  kiosk?: KioskCredentials,
): Promise<T> => {
  // A network failure or timeout throws a plain error here, which callers treat as "offline".
  // Without the timeout a hung server would leave the employee waiting with no result.
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  let res: Response;
  let json: any;
  try {
    res = await fetch(`${API_URL}${path}`, {
      method,
      headers: {
        'Content-Type': 'application/json',
        ...(kiosk && { 'X-Kiosk-Id': kiosk.kioskId, 'X-Kiosk-Key': kiosk.apiKey }),
      },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: controller.signal,
    });
    json = await res.json().catch(() => null);
  } catch (e) {
    throw new TypeError(controller.signal.aborted ? 'The server took too long to respond' : String(e));
  } finally {
    clearTimeout(timer);
  }
  if (res.status === 401) throw new UnauthorizedError(json?.message ?? 'Unauthorized');
  if (res.status >= 400 && res.status < 500) throw new RejectedError(json?.message ?? `Request failed (${res.status})`);
  if (!res.ok) throw new Error(json?.message ?? `Request failed (${res.status})`);
  return json.data as T;
};

export const pairKiosk = async (code: string): Promise<KioskCredentials> => {
  const kiosk = await request<KioskCredentials>('POST', '/kiosk-device/pair', { code });
  await saveCredentials(kiosk);
  return kiosk;
};

const utcOffsetMinutes = () => -new Date().getTimezoneOffset();

export const heartbeat = (kiosk: KioskCredentials) =>
  request<{ time12Hour: string; date: string }>(
    'POST',
    `/kiosk-device/heartbeat?utcOffsetMinutes=${utcOffsetMinutes()}`,
    undefined,
    kiosk,
  );

export type Directory = { employees: Employee[]; templates: FaceTemplate[] };

/** Last synced employees and face templates, so the kiosk recognizes people while offline. */
export const loadDirectory = async (): Promise<Directory> => {
  const stored = await AsyncStorage.getItem(DIRECTORY_KEY);
  return stored ? JSON.parse(stored) : { employees: [], templates: [] };
};

// shortcut: full download each time, fine for a few hundred employees; switch to `?since=` deltas when orgs grow.
export const syncDirectory = async (kiosk: KioskCredentials): Promise<Directory> => {
  const [employees, { templates }] = await Promise.all([
    request<Employee[]>('GET', '/face-template/employees', undefined, kiosk),
    request<{ templates: FaceTemplate[] }>('GET', '/face-template/sync', undefined, kiosk),
  ]);
  const directory = {
    employees,
    templates: templates.filter((t) => t.modelName === FACE_MODEL.modelName),
  };
  await AsyncStorage.setItem(DIRECTORY_KEY, JSON.stringify(directory));
  return directory;
};

export const unlockEnrollment = (kiosk: KioskCredentials, code: string) =>
  request<{ expiresInSeconds: number }>('POST', '/kiosk-device/enrollment/unlock', { code }, kiosk);

export const uploadTemplate = (kiosk: KioskCredentials, employeeId: string, embedding: number[]) =>
  request('POST', '/face-template/enroll', { employeeId, embedding, ...FACE_MODEL }, kiosk);

const readQueue = async (): Promise<Punch[]> => JSON.parse((await AsyncStorage.getItem(QUEUE_KEY)) ?? '[]');

/**
 * Records a punch. Returns null when the kiosk is offline: the punch is then kept on the
 * device and replayed by flushPunches, and its event id stops the server double-counting it.
 */
export const punch = async (
  kiosk: KioskCredentials,
  employeeId: string,
  faceMatchScore: number,
): Promise<PunchResult | null> => {
  const entry: Punch = {
    employeeId,
    occurredAt: new Date().toISOString(),
    utcOffsetMinutes: utcOffsetMinutes(),
    externalEventId: `${kiosk.kioskId}-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`,
    // Similarity is cosine (-1..1); the server stores 0..1.
    faceMatchScore: Math.max(0, Math.min(1, faceMatchScore)),
    // Liveness runs before a match is reported, so every match passed it.
    livenessPassed: true,
  };
  try {
    return await request<PunchResult>('POST', '/attendance/kiosk', entry, kiosk);
  } catch (e) {
    if (e instanceof RejectedError || e instanceof UnauthorizedError) throw e;
    await AsyncStorage.setItem(QUEUE_KEY, JSON.stringify([...(await readQueue()), entry]));
    return null;
  }
};

/** Replays punches recorded while offline, oldest first. Stops at the first network failure. */
export const flushPunches = async (kiosk: KioskCredentials) => {
  const done = new Set<string>();
  for (const entry of await readQueue()) {
    // Punches made while this phone was a different kiosk belong to that kiosk's organization.
    if (!entry.externalEventId.startsWith(`${kiosk.kioskId}-`)) {
      console.warn('Dropping offline punch from a previous pairing', entry.externalEventId);
      done.add(entry.externalEventId);
      continue;
    }
    try {
      await request('POST', '/attendance/kiosk', entry, kiosk);
    } catch (e) {
      if (!(e instanceof RejectedError)) break;
      // Refused for good (e.g. employee removed or on leave); keeping it would block the queue forever.
      console.warn('Dropping rejected offline punch', entry.externalEventId, e.message);
    }
    done.add(entry.externalEventId);
  }
  if (done.size === 0) return;
  // Re-read so punches queued while this loop ran are kept.
  const remaining = (await readQueue()).filter((entry) => !done.has(entry.externalEventId));
  await AsyncStorage.setItem(QUEUE_KEY, JSON.stringify(remaining));
};
