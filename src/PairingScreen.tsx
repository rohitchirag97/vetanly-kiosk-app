import { useState } from 'react';
import { ActivityIndicator, Image, Pressable, StyleSheet, Text, TextInput, View } from 'react-native';

import { type KioskCredentials, pairKiosk } from './kiosk';
import { colors } from './ui';

export function PairingScreen({ onPaired }: { onPaired: (kiosk: KioskCredentials) => void }) {
  const [code, setCode] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async () => {
    if (code.length !== 6 || busy) return;
    setBusy(true);
    setError(null);
    try {
      onPaired(await pairKiosk(code));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not reach the server');
      setCode('');
    } finally {
      setBusy(false);
    }
  };

  return (
    <View style={styles.container}>
      <Image source={require('./assets/vetanly-icon.png')} style={styles.logo} accessibilityIgnoresInvertColors />
      <Text style={styles.title}>Pair this kiosk</Text>
      <Text style={styles.hint}>
        In the Vetanly dashboard, open Kiosk devices and tap "Pair device". Enter the 6-digit code shown there.
      </Text>
      <TextInput
        accessibilityLabel="Pairing code"
        value={code}
        onChangeText={(text) => setCode(text.replace(/\D/g, '').slice(0, 6))}
        onSubmitEditing={submit}
        keyboardType="number-pad"
        maxLength={6}
        autoFocus
        style={styles.input}
      />
      {error && <Text accessibilityRole="alert" style={styles.error}>{error}</Text>}
      <Pressable
        accessibilityRole="button"
        accessibilityState={{ disabled: code.length !== 6 || busy }}
        onPress={submit}
        style={[styles.button, (code.length !== 6 || busy) && styles.disabled]}
      >
        {busy ? <ActivityIndicator color="#fff" /> : <Text style={styles.buttonText}>Pair</Text>}
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 32, gap: 20 },
  logo: { width: 72, height: 72 },
  title: { fontSize: 32, fontWeight: '700' },
  hint: { fontSize: 18, textAlign: 'center', maxWidth: 520, opacity: 0.7 },
  input: {
    fontSize: 40,
    letterSpacing: 16,
    textAlign: 'center',
    width: 320,
    borderWidth: 2,
    borderRadius: 12,
    borderColor: '#999',
    paddingVertical: 12,
  },
  error: { color: '#c62828', fontSize: 16 },
  button: { backgroundColor: colors.brand, borderRadius: 12, paddingVertical: 16, width: 320, alignItems: 'center' },
  disabled: { opacity: 0.5 },
  buttonText: { color: '#fff', fontSize: 20, fontWeight: '600' },
});
