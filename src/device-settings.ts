import AsyncStorage from '@react-native-async-storage/async-storage';
import { Vibration } from 'react-native';

import KioskDevice from '../specs/NativeKioskDevice';

export type KioskSettings = {
  keepAwake: boolean;
  showOverLockScreen: boolean;
  pinned: boolean;
  requireAutoTime: boolean;
  showNetwork: boolean;
  sound: boolean;
  vibrate: boolean;
};

// Device controls start off so each organization opts in; punch feedback starts on.
export const DEFAULT_SETTINGS: KioskSettings = {
  keepAwake: false,
  showOverLockScreen: false,
  pinned: false,
  requireAutoTime: false,
  showNetwork: false,
  sound: true,
  vibrate: true,
};

const SETTINGS_KEY = 'kiosk-settings';

export const loadSettings = async (): Promise<KioskSettings> => {
  const stored = await AsyncStorage.getItem(SETTINGS_KEY);
  return { ...DEFAULT_SETTINGS, ...(stored ? JSON.parse(stored) : {}) };
};

export const saveSettings = (settings: KioskSettings) =>
  AsyncStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));

/** Pushes the switches that change the phone's behaviour to the native side. */
export const applySettings = (settings: KioskSettings) => {
  KioskDevice?.setKeepAwake(settings.keepAwake);
  KioskDevice?.setShowOverLockScreen(settings.showOverLockScreen);
  KioskDevice?.setPinned(settings.pinned);
};

/** Tells the person the punch went through (or did not) without reading the screen. */
export const punchFeedback = (settings: KioskSettings, success: boolean) => {
  if (settings.sound) KioskDevice?.playTone(success ? 'success' : 'error');
  if (settings.vibrate) Vibration.vibrate(success ? [0, 60, 80, 60] : 400);
};

export { KioskDevice };
export type { DeviceStatus } from '../specs/NativeKioskDevice';
