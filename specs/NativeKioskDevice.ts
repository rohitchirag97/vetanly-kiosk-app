import type { TurboModule } from 'react-native';
import { TurboModuleRegistry } from 'react-native';

export type DeviceStatus = {
  autoTime: boolean;
  autoTimeZone: boolean;
  /** A PIN, pattern or password lock is set on the phone. */
  deviceSecure: boolean;
  pinned: boolean;
  /** 'wifi' | 'cellular' | 'ethernet' | 'other' | 'none' */
  networkType: string;
  /** 0 to 4 bars, or -1 when the phone does not report it. */
  signalLevel: number;
};

export interface Spec extends TurboModule {
  setKeepAwake(enabled: boolean): void;
  setShowOverLockScreen(enabled: boolean): void;
  setPinned(enabled: boolean): void;
  getStatus(): Promise<DeviceStatus>;
  /** 'date' | 'security' | 'wifi' */
  openSettings(screen: string): void;
  /** Short confirmation tone: 'success' or 'error'. Plays at media volume. */
  playTone(kind: string): void;
}

// Android only; null elsewhere so the kiosk still runs without these controls.
export default TurboModuleRegistry.get<Spec>('KioskDevice');
