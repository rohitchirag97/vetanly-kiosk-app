// Mirrors EnrollmentCodec.cpp in the face module: a 16-byte header (magic, version,
// sample count, embedding length as little-endian uint32) followed by float32 samples.
// The server stores each sample as its own face template row.
const MAGIC = 0x45464352;
const VERSION = 1;
const HEADER_BYTES = 16;

export const encodeEnrollment = (samples: number[][]): ArrayBuffer => {
  const length = samples[0].length;
  const buffer = new ArrayBuffer(HEADER_BYTES + samples.length * length * 4);
  const view = new DataView(buffer);
  view.setUint32(0, MAGIC, true);
  view.setUint32(4, VERSION, true);
  view.setUint32(8, samples.length, true);
  view.setUint32(12, length, true);
  samples.forEach((sample, i) => {
    sample.forEach((value, j) => view.setFloat32(HEADER_BYTES + (i * length + j) * 4, value, true));
  });
  return buffer;
};

export const decodeEnrollment = (buffer: ArrayBuffer): number[][] => {
  const view = new DataView(buffer);
  if (view.getUint32(0, true) !== MAGIC || view.getUint32(4, true) !== VERSION) {
    throw new Error('Unsupported enrollment format');
  }
  const count = view.getUint32(8, true);
  const length = view.getUint32(12, true);
  return Array.from({ length: count }, (_, i) =>
    Array.from({ length }, (__, j) => view.getFloat32(HEADER_BYTES + (i * length + j) * 4, true)),
  );
};
