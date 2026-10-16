/**
 * @format
 */

import ReactTestRenderer from 'react-test-renderer';
import * as Keychain from 'react-native-keychain';
import App from '../App';

jest.mock('react-native-safe-area-context', () => require('react-native-safe-area-context/jest/mock').default);
jest.mock('@react-native-async-storage/async-storage', () => ({
  getItem: jest.fn(async () => null),
  setItem: jest.fn(),
  removeItem: jest.fn(),
}));
// The camera and face engine are native-only; the kiosk screen is covered on a device.
jest.mock('../src/KioskScreen', () => ({
  KioskScreen: ({ kiosk }: { kiosk: { name: string } }) => {
    const { Text } = require('react-native');
    return <Text>{kiosk.name}</Text>;
  },
}));
jest.mock('react-native-keychain', () => ({
  ACCESSIBLE: { AFTER_FIRST_UNLOCK_THIS_DEVICE_ONLY: 'x' },
  getGenericPassword: jest.fn(),
  setGenericPassword: jest.fn(),
  resetGenericPassword: jest.fn(),
}));

const stored = { kioskId: 'k1', name: 'Front desk', apiKey: 'secret' };
const respond = (status: number, data: unknown = null) =>
  (globalThis.fetch = jest.fn(async () => ({ status, ok: status < 400, json: async () => ({ data }) })) as any);

const render = async () => {
  let tree!: ReactTestRenderer.ReactTestRenderer;
  await ReactTestRenderer.act(async () => {
    tree = ReactTestRenderer.create(<App />);
  });
  return tree;
};

const texts = (tree: ReactTestRenderer.ReactTestRenderer) =>
  tree.root.findAllByType(require('react-native').Text).map((t) => [t.props.children].flat().join(''));

beforeEach(() => jest.clearAllMocks());

test('shows pairing when no credentials are stored', async () => {
  (Keychain.getGenericPassword as jest.Mock).mockResolvedValue(false);
  expect(texts(await render())).toContain('Pair this kiosk');
});

test('stays paired while the server accepts the key', async () => {
  (Keychain.getGenericPassword as jest.Mock).mockResolvedValue({ password: JSON.stringify(stored) });
  respond(200, { time12Hour: '9:00 AM' });
  const tree = await render();
  expect(texts(tree)).toContain('Front desk');
  expect(globalThis.fetch).toHaveBeenCalledWith(
    expect.stringContaining('/kiosk-device/heartbeat'),
    expect.objectContaining({ headers: expect.objectContaining({ 'X-Kiosk-Id': 'k1', 'X-Kiosk-Key': 'secret' }) }),
  );
  await ReactTestRenderer.act(async () => tree.unmount());
});

test('a revoked key clears credentials and returns to pairing', async () => {
  (Keychain.getGenericPassword as jest.Mock).mockResolvedValue({ password: JSON.stringify(stored) });
  respond(401);
  const tree = await render();
  expect(Keychain.resetGenericPassword).toHaveBeenCalled();
  expect(texts(tree)).toContain('Pair this kiosk');
});
