import { ScrollView, StyleSheet, Switch, Text, View } from 'react-native';

import { type DeviceStatus, KioskDevice, type KioskSettings } from './device-settings';
import { Button, colors, ui } from './ui';

type Row = {
  key: keyof KioskSettings;
  label: string;
  description: string;
  status?: (s: DeviceStatus) => string;
  action?: { label: string; screen: string };
};

const ROWS: Row[] = [
  {
    key: 'keepAwake',
    label: 'Keep screen awake',
    description: 'The screen never turns off while the kiosk is open. Keep the phone on its charger.',
  },
  {
    key: 'showOverLockScreen',
    label: 'Open over the lock screen',
    description: 'The kiosk shows on top of the lock screen, so nobody has to unlock the phone.',
    status: (s) =>
      s.deviceSecure
        ? 'This phone has a screen lock. After a restart, unlock it once so the kiosk can sign in.'
        : 'No screen lock is set on this phone.',
    action: { label: 'Screen lock settings', screen: 'security' },
  },
  {
    key: 'pinned',
    label: 'Lock to this app',
    description: "Pins the kiosk so people can't leave it or open other apps. Android asks to confirm.",
    status: (s) => (s.pinned ? 'The kiosk is pinned.' : 'The kiosk is not pinned.'),
  },
  {
    key: 'requireAutoTime',
    label: 'Require automatic date & time',
    description: "Warns on the kiosk when the phone's date, time or time zone isn't set automatically. Punches use the phone's clock.",
    status: (s) =>
      s.autoTime && s.autoTimeZone
        ? 'Date, time and time zone are automatic.'
        : `${!s.autoTime ? 'Date & time' : 'Time zone'} is set manually.`,
    action: { label: 'Date & time settings', screen: 'date' },
  },
  {
    key: 'sound',
    label: 'Sound on punch',
    description: 'A short beep when attendance is marked, and a different tone when it is refused. Uses the media volume.',
  },
  {
    key: 'vibrate',
    label: 'Vibrate on punch',
    description: 'A short buzz when attendance is marked, and a longer one when it is refused.',
  },
  {
    key: 'showNetwork',
    label: 'Show network strength',
    description: 'Shows Wi-Fi or mobile signal bars at the top of the kiosk.',
    status: (s) => networkLabel(s),
    action: { label: 'Wi-Fi settings', screen: 'wifi' },
  },
];

export const networkLabel = (s: DeviceStatus) => {
  const name =
    { wifi: 'Wi-Fi', cellular: 'Mobile data', ethernet: 'Ethernet', none: 'No connection' }[s.networkType] ?? 'Connected';
  return s.signalLevel >= 0 && s.networkType !== 'none' ? `${name}, ${s.signalLevel} of 4 bars` : name;
};

export function SettingsPanel({
  settings,
  status,
  onChange,
  onClose,
}: {
  settings: KioskSettings;
  status: DeviceStatus | null;
  onChange: (next: KioskSettings) => void;
  onClose: () => void;
}) {
  return (
    <View style={[ui.panel, ui.tall]}>
      <Text style={ui.title}>Kiosk settings</Text>
      <ScrollView contentContainerStyle={styles.list}>
        {!KioskDevice && <Text style={ui.error}>These settings are only available on Android.</Text>}
        {ROWS.map((row) => (
          <View key={row.key} style={styles.row}>
            <View style={styles.header}>
              <Text style={styles.label}>{row.label}</Text>
              <Switch
                accessibilityLabel={row.label}
                value={settings[row.key]}
                trackColor={{ false: colors.border, true: colors.brand }}
                thumbColor="#fff"
                onValueChange={(value) => onChange({ ...settings, [row.key]: value })}
              />
            </View>
            <Text style={ui.hint}>{row.description}</Text>
            {status && row.status && <Text style={styles.status}>{row.status(status)}</Text>}
            {row.action && KioskDevice && (
              <Text
                accessibilityRole="link"
                style={[styles.link, status?.pinned && styles.linkDisabled]}
                onPress={status?.pinned ? undefined : () => KioskDevice?.openSettings(row.action!.screen)}
              >
                {row.action.label}
              </Text>
            )}
          </View>
        ))}
        {status?.pinned && (
          <Text style={ui.hint}>Turn off "Lock to this app" to open the phone's settings.</Text>
        )}
      </ScrollView>
      <View style={ui.row}>
        <Button label="Done" onPress={onClose} />
      </View>
    </View>
  );
}

/** Four signal bars for the kiosk's top bar. */
export function NetworkBars({ status }: { status: DeviceStatus }) {
  const level = status.networkType === 'none' ? 0 : status.signalLevel;
  return (
    <View accessible accessibilityLabel={networkLabel(status)} style={styles.bars}>
      {[1, 2, 3, 4].map((bar) => (
        <View key={bar} style={[styles.bar, { height: 4 + bar * 4 }, bar <= level && styles.barOn]} />
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  list: { paddingBottom: 8 },
  row: { gap: 4, paddingVertical: 16, borderBottomWidth: StyleSheet.hairlineWidth, borderColor: colors.border },
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: 12 },
  label: { flex: 1, fontSize: 17, fontWeight: '600', color: colors.text },
  status: { fontSize: 14, color: colors.success, fontWeight: '500' },
  link: { fontSize: 15, color: colors.brand, fontWeight: '700', paddingVertical: 4 },
  linkDisabled: { color: '#999' },
  bars: { flexDirection: 'row', alignItems: 'flex-end', gap: 2, height: 20 },
  bar: { width: 4, borderRadius: 1, backgroundColor: 'rgba(255,255,255,0.3)' },
  barOn: { backgroundColor: '#fff' },
});
