import * as Keychain from 'react-native-keychain';
import { flushPunches, loadCredentials, punch, RejectedError } from '../src/kiosk';

jest.mock('react-native-keychain', () => ({
  ACCESSIBLE: { AFTER_FIRST_UNLOCK_THIS_DEVICE_ONLY: 'x' },
  getGenericPassword: jest.fn(),
  setGenericPassword: jest.fn(),
  resetGenericPassword: jest.fn(),
}));
jest.mock('@react-native-async-storage/async-storage', () => {
  const store = new Map<string, string>();
  return {
    getItem: jest.fn(async (key: string) => store.get(key) ?? null),
    setItem: jest.fn(async (key: string, value: string) => {
      store.set(key, value);
    }),
    removeItem: jest.fn(async (key: string) => {
      store.delete(key);
    }),
    clear: () => store.clear(),
  };
});

const storage = require('@react-native-async-storage/async-storage');
const kiosk = { kioskId: 'k1', name: 'Gate', apiKey: 'secret' };
const queue = async () => JSON.parse((await storage.getItem('kiosk-punch-queue')) ?? '[]');
const reply = (status: number, data: unknown = null) => ({ status, ok: status < 400, json: async () => ({ data, message: 'On leave' }) });

beforeEach(() => {
  storage.clear();
  jest.useRealTimers();
});

test('an offline punch is kept on the device and replayed with the same event id', async () => {
  globalThis.fetch = jest.fn().mockRejectedValue(new TypeError('Network request failed')) as any;
  expect(await punch(kiosk, 'emp-1', 0.8)).toBeNull();
  const [saved] = await queue();
  expect(saved.employeeId).toBe('emp-1');

  globalThis.fetch = jest.fn().mockResolvedValue(reply(201, { eventType: 'check_in' })) as any;
  await flushPunches(kiosk);
  expect(JSON.parse((globalThis.fetch as jest.Mock).mock.calls[0][1].body).externalEventId).toBe(saved.externalEventId);
  expect(await queue()).toEqual([]);
});

test('a server that never answers counts as offline instead of hanging', async () => {
  jest.useFakeTimers();
  globalThis.fetch = jest.fn(
    (_url: string, init: { signal: AbortSignal }) =>
      new Promise((_, reject) => init.signal.addEventListener('abort', () => reject(new Error('Aborted')))),
  ) as any;
  const result = punch(kiosk, 'emp-2', 0.9);
  await jest.advanceTimersByTimeAsync(8_000);
  expect(await result).toBeNull();
  expect(await queue()).toHaveLength(1);
});

test('a refused punch is shown, not queued', async () => {
  globalThis.fetch = jest.fn().mockResolvedValue(reply(409)) as any;
  await expect(punch(kiosk, 'emp-3', 0.9)).rejects.toBeInstanceOf(RejectedError);
  expect(await queue()).toEqual([]);
});

test('punches saved under a previous pairing are discarded, not sent to the new kiosk', async () => {
  globalThis.fetch = jest.fn().mockRejectedValue(new TypeError('offline')) as any;
  await punch(kiosk, 'old-org-employee', 0.9);
  globalThis.fetch = jest.fn().mockResolvedValue(reply(201)) as any;
  await flushPunches({ ...kiosk, kioskId: 'k2' });
  expect(globalThis.fetch).not.toHaveBeenCalled();
  expect(await queue()).toEqual([]);
});

test('replay stops at the first network failure and keeps the rest', async () => {
  globalThis.fetch = jest.fn().mockRejectedValue(new TypeError('offline')) as any;
  await punch(kiosk, 'a', 0.9);
  await punch(kiosk, 'b', 0.9);
  globalThis.fetch = jest
    .fn()
    .mockResolvedValueOnce(reply(201))
    .mockRejectedValueOnce(new TypeError('offline')) as any;
  await flushPunches(kiosk);
  expect((await queue()).map((p: { employeeId: string }) => p.employeeId)).toEqual(['b']);
});

test('a kiosk paired before the rebrand keeps its key under the new name', async () => {
  (Keychain.getGenericPassword as jest.Mock).mockImplementation(async ({ service }: { service: string }) =>
    service === 'hazripro-kiosk' ? { password: JSON.stringify(kiosk) } : false,
  );
  expect(await loadCredentials()).toEqual(kiosk);
  expect(Keychain.setGenericPassword).toHaveBeenCalledWith('k1', JSON.stringify(kiosk), expect.objectContaining({ service: 'vetanly-kiosk' }));
  expect(Keychain.resetGenericPassword).toHaveBeenCalledWith({ service: 'hazripro-kiosk' });
});
