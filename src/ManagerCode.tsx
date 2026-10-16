import { useState } from 'react';
import { StyleSheet, Text, TextInput, View } from 'react-native';

import { type KioskCredentials, unlockEnrollment } from './kiosk';
import { Button, errorMessage, ui } from './ui';

/** Manager-only screens (Enroll, Settings) open with the code issued in the dashboard. */
export function ManagerCode({
  kiosk,
  onUnlocked,
  onClose,
}: {
  kiosk: KioskCredentials;
  onUnlocked: () => void;
  onClose: () => void;
}) {
  const [code, setCode] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    if (code.length !== 6 || busy) return;
    setBusy(true);
    setError(null);
    try {
      await unlockEnrollment(kiosk, code);
      onUnlocked();
    } catch (e) {
      setError(errorMessage(e));
      setCode('');
    } finally {
      setBusy(false);
    }
  };

  return (
    <View style={ui.panel}>
      <Text style={ui.title}>Manager code</Text>
      <Text style={ui.hint}>
        In the Vetanly dashboard, open Kiosk devices and tap "Manager code" for this kiosk.
      </Text>
      <TextInput
        accessibilityLabel="Manager code"
        value={code}
        onChangeText={(text) => setCode(text.replace(/\D/g, '').slice(0, 6))}
        onSubmitEditing={submit}
        keyboardType="number-pad"
        maxLength={6}
        autoFocus
        style={styles.codeInput}
      />
      {error && <Text accessibilityRole="alert" style={ui.error}>{error}</Text>}
      <View style={ui.row}>
        <Button label="Cancel" onPress={onClose} secondary />
        <Button label={busy ? 'Checking…' : 'Unlock'} onPress={submit} disabled={code.length !== 6 || busy} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  codeInput: {
    fontSize: 36,
    letterSpacing: 14,
    textAlign: 'center',
    borderWidth: 2,
    borderRadius: 12,
    borderColor: '#999',
    paddingVertical: 10,
    color: '#111',
  },
});
