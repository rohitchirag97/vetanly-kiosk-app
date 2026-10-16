# Vetanly Kiosk

An Android app that turns a phone or tablet into a face-recognition attendance kiosk for
[Vetanly](https://vetanly.com). Employees look at the camera to check in or out, with no card,
PIN or touch needed.

Face recognition runs **on the device** (detection, liveness check and matching), using a
recognizer vendored from Margelo's
[face-recognition-demo](https://margelo.com/blog/on-device-face-recognition-react-native) (MIT).
Only face templates (numbers, not photos) and attendance punches go to the Vetanly server.

Built with React Native 0.87 (New Architecture), VisionCamera 5, Nitro Modules and
react-native-worklets.

## How it works

- **Pairing.** In the Vetanly dashboard, go to **Kiosks** and create a kiosk. It shows a pairing
  code. Enter the code on the kiosk. The kiosk receives its own API key and stores it in the
  Android Keystore, so no one has to sign in on the device. Unpairing a kiosk, or rotating its
  key in the dashboard, logs the device out.
- **Status.** The kiosk sends a regular heartbeat, so the dashboard shows each kiosk as
  Online, Offline or Not connected.
- **Enrolling faces.** Tap **Enroll** and enter a manager's 6-digit enrollment code (issued in the
  dashboard). Pick the employee and capture 5 samples. Enrolled faces sync to every kiosk in the
  organization, so each person enrolls once, not once per kiosk.
- **Punching.** The camera runs all the time. A recognized face with a live-person check records
  the next punch automatically: check-in, then check-out.
  - Repeat scans are ignored for **15 minutes** after a punch, so walking past the kiosk does not
    punch again. The server applies the same rule.
  - A check-out after that window moves the check-out time later, which records overtime.
  - Night shifts that cross midnight are handled by the server: a punch belongs to the nearest
    shift, not the calendar date.
- **Offline.** Employees and face templates are cached on the device (refreshed every 5 minutes),
  so recognition keeps working without internet. Punches made offline are queued as
  *Saved offline* and sent, oldest first, when the connection returns. Each punch carries a unique
  ID, so a retry is never recorded twice.
- **Feedback.** A success or error tone and vibration after each punch (each can be switched off).

### Settings (manager code required)

| Setting | What it does |
|---|---|
| Keep screen awake | The screen never turns off. |
| Show over lock screen | The kiosk opens above the lock screen after a reboot or power cut. |
| Pin app | Uses Android screen pinning so people cannot leave the app. |
| Require automatic date & time | Warns when the phone's clock is set by hand, so punch times can be trusted. |
| Show network | Shows Wi-Fi or mobile signal strength on screen. |
| Sound / Vibration | Punch feedback. |

Device controls start **off** so each organization opts in; sound and vibration start on.
The app always stays in portrait.

## Requirements

- Node.js 22.11 or later and [Bun](https://bun.sh) (the repo uses `bun.lock` and Bun workspaces).
- JDK 17 and the Android SDK (via Android Studio). See React Native's
  [environment setup](https://reactnative.dev/docs/set-up-your-environment).
- An Android phone or tablet with Android 7.0 (API 24) or later and a front camera.
  A mid-range phone or better gives the fastest recognition.
- The Vetanly server running and reachable from the phone (see `../server`).

iOS is not supported yet. The native kiosk controls (`KioskDevice`) are Android only.

## Run on a phone

1. **Point the app at your server.** Edit `src/config.ts` and set `API_URL` to an address the
   phone can reach. During development that is your computer's LAN IP, for example
   `http://192.168.0.178:8000/api/v1`. `localhost` will not work on a phone.
2. **Install dependencies**

   ```sh
   bun install
   ```

3. **Connect the phone.** Turn on *Developer options* and *USB debugging*, connect it by USB and
   check that it shows up:

   ```sh
   adb devices
   ```

4. **Start Metro** (keep this terminal open):

   ```sh
   bun start
   ```

5. **Build and install** in a second terminal:

   ```sh
   bun run android
   ```

6. If the app shows a red screen saying it cannot reach the development server, forward Metro's
   port and reload:

   ```sh
   adb reverse tcp:8081 tcp:8081
   ```

7. Pair the kiosk with a code from the dashboard.

The phone and computer must be on the same network, and Windows Firewall must allow port 8000
(the API) for the kiosk to reach a local server.

## Setting up the device for kiosk use

- Plug it in permanently and mount it at face height with even, front-facing light.
  Strong light behind the person makes recognition fail.
- Turn on **Keep screen awake**, **Show over lock screen** and **Pin app** in the kiosk settings.
- Set the phone's date, time and time zone to **automatic**, then turn on
  **Require automatic date & time**.
- Keep Wi-Fi on. Offline punches are saved, but faces enrolled on other kiosks only arrive
  after a sync.
- Turn off battery optimization for the app so Android does not stop it.

## Project layout

```
App.tsx                  Pairing vs. kiosk screen, settings load
src/PairingScreen.tsx    Enter the pairing code
src/KioskScreen.tsx      Camera, recognition, punching, cooldown, sync
src/EnrollFlow.tsx       Manager enrollment (5 samples per employee)
src/ManagerCode.tsx      6-digit code that unlocks Enroll and Settings
src/SettingsPanel.tsx    Device and feedback switches
src/useFaceEngine.ts     Detection, liveness and matching thresholds
src/kiosk.ts             API client, Keychain credentials, offline cache and punch queue
src/device-settings.ts   Settings storage and the native KioskDevice bridge
specs/                   TurboModule spec for KioskDevice (codegen)
android/.../KioskDeviceModule.kt   Native keep-awake, pinning, lock screen, network, tones
modules/react-native-face-recognition   Vendored Margelo face recognizer (Nitro module)
```

## Scripts

```sh
bun start          # Metro dev server
bun run android    # build and install the debug app
bun run lint       # ESLint
bunx tsc --noEmit  # typecheck
bun run test       # Jest
```

## Before a release

- Move `API_URL` in `src/config.ts` from the hard-coded LAN IP into build configuration.
- Create a release keystore and signing config. `android/app/debug.keystore` is the standard
  debug key; never commit the release keystore or its passwords.
- Remove the legacy `hazripro-kiosk` Keychain fallback in `src/kiosk.ts` once every kiosk has
  run a post-rebrand build.

## Troubleshooting

- **Network request failed when pairing.** `API_URL` is wrong or the phone cannot reach it. Open
  the URL in the phone's browser to check, and allow port 8000 through the firewall.
- **Error 500 from the development server, or "Failed to get NitroModules".** Stop Metro, then run
  `bun start --reset-cache` and rebuild with `bun run android`.
- **Poor or slow recognition.** Improve the lighting, clean the camera, and re-enroll the person
  in the same light the kiosk will be used in.
- **Logs.** `adb logcat *:S ReactNativeJS:V AndroidRuntime:E` shows JavaScript logs and crashes.
