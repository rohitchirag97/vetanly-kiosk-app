import { useCallback, useEffect, useState } from 'react';
import { ActivityIndicator, StatusBar, StyleSheet } from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';

import {
  clearCredentials,
  flushPunches,
  heartbeat,
  type KioskCredentials,
  loadCredentials,
  UnauthorizedError,
} from './src/kiosk';
import { KioskScreen } from './src/KioskScreen';
import { PairingScreen } from './src/PairingScreen';

const HEARTBEAT_MS = 60_000;

type Session = { status: 'loading' } | { status: 'unpaired' } | { status: 'paired'; kiosk: KioskCredentials };

function App() {
  const [session, setSession] = useState<Session>({ status: 'loading' });
  const [online, setOnline] = useState(true);

  const unpair = useCallback(async () => {
    await clearCredentials();
    setSession({ status: 'unpaired' });
  }, []);

  // A revoked or rotated key sends the kiosk back to pairing; network errors keep it running offline.
  const check = useCallback(
    async (kiosk: KioskCredentials) => {
      try {
        await heartbeat(kiosk);
        setOnline(true);
        await flushPunches(kiosk);
      } catch (e) {
        if (e instanceof UnauthorizedError) await unpair();
        else setOnline(false);
      }
    },
    [unpair],
  );

  useEffect(() => {
    loadCredentials()
      .catch(() => null)
      .then((kiosk) => setSession(kiosk ? { status: 'paired', kiosk } : { status: 'unpaired' }));
  }, []);

  const kiosk = session.status === 'paired' ? session.kiosk : null;
  useEffect(() => {
    if (!kiosk) return;
    check(kiosk);
    const timer = setInterval(() => check(kiosk), HEARTBEAT_MS);
    return () => clearInterval(timer);
  }, [kiosk, check]);

  return (
    <SafeAreaProvider>
      <StatusBar barStyle={kiosk ? 'light-content' : 'dark-content'} />
      {session.status === 'paired' ? (
        // Full-bleed camera; the kiosk screen pads its own overlays with the safe-area insets.
        <KioskScreen kiosk={session.kiosk} online={online} onUnauthorized={unpair} />
      ) : (
        <SafeAreaView style={styles.screen}>
          {session.status === 'loading' ? (
            <ActivityIndicator style={styles.screen} size="large" />
          ) : (
            <PairingScreen onPaired={(paired) => setSession({ status: 'paired', kiosk: paired })} />
          )}
        </SafeAreaView>
      )}
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: '#fff' },
});

export default App;
