import { decodeEnrollment, encodeEnrollment } from '../src/enrollment-buffer';

test('round-trips samples through the native enrollment format', () => {
  const samples = [
    [0.5, -0.25, 1],
    [0.125, 0, -1],
  ];
  const buffer = encodeEnrollment(samples);
  expect(buffer.byteLength).toBe(16 + 6 * 4);
  expect(new DataView(buffer).getUint32(0, true)).toBe(0x45464352);
  expect(decodeEnrollment(buffer)).toEqual(samples);
});

test('rejects foreign buffers', () => {
  expect(() => decodeEnrollment(new ArrayBuffer(16))).toThrow('Unsupported enrollment format');
});
