package com.kioskapp

import android.app.Activity
import android.app.ActivityManager
import android.app.KeyguardManager
import android.content.Intent
import android.media.AudioManager
import android.media.ToneGenerator
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.net.wifi.WifiManager
import android.os.Build
import android.provider.Settings
import android.telephony.TelephonyManager
import android.view.WindowManager
import com.facebook.react.BaseReactPackage
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.UiThreadUtil
import com.facebook.react.module.model.ReactModuleInfo
import com.facebook.react.module.model.ReactModuleInfoProvider
import com.kioskapp.specs.NativeKioskDeviceSpec

/** Device controls an organization can switch on for a dedicated attendance kiosk. */
class KioskDeviceModule(private val context: ReactApplicationContext) : NativeKioskDeviceSpec(context) {

  private fun onActivity(block: (Activity) -> Unit) {
    val activity = context.currentActivity ?: return
    UiThreadUtil.runOnUiThread { block(activity) }
  }

  override fun setKeepAwake(enabled: Boolean) = onActivity {
    if (enabled) it.window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    else it.window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
  }

  override fun setShowOverLockScreen(enabled: Boolean) = onActivity {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
      it.setShowWhenLocked(enabled)
      it.setTurnScreenOn(enabled)
    } else {
      @Suppress("DEPRECATION")
      val flags = WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED or WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON
      if (enabled) it.window.addFlags(flags) else it.window.clearFlags(flags)
    }
  }

  // Screen pinning: Android asks the person to confirm. A device-owner setup would skip the prompt.
  override fun setPinned(enabled: Boolean) = onActivity {
    try {
      if (enabled && !isPinned()) it.startLockTask() else if (!enabled && isPinned()) it.stopLockTask()
    } catch (e: Exception) {
      // Pinning can be refused (e.g. disabled by the OEM); the setting then simply has no effect.
    }
  }

  private fun isPinned() =
    context.getSystemService(ActivityManager::class.java).lockTaskModeState != ActivityManager.LOCK_TASK_MODE_NONE

  override fun getStatus(promise: Promise) {
    try {
      val resolver = context.contentResolver
      val (networkType, signalLevel) = network()
      promise.resolve(
        Arguments.createMap().apply {
          putBoolean("autoTime", Settings.Global.getInt(resolver, Settings.Global.AUTO_TIME, 1) == 1)
          putBoolean("autoTimeZone", Settings.Global.getInt(resolver, Settings.Global.AUTO_TIME_ZONE, 1) == 1)
          putBoolean("deviceSecure", context.getSystemService(KeyguardManager::class.java).isDeviceSecure)
          putBoolean("pinned", isPinned())
          putString("networkType", networkType)
          putInt("signalLevel", signalLevel)
        },
      )
    } catch (e: Exception) {
      promise.reject("status_failed", e)
    }
  }

  private fun network(): Pair<String, Int> {
    val connectivity = context.getSystemService(ConnectivityManager::class.java)
    val caps = connectivity.getNetworkCapabilities(connectivity.activeNetwork) ?: return "none" to 0
    return when {
      caps.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) -> {
        @Suppress("DEPRECATION")
        val rssi = context.applicationContext.getSystemService(WifiManager::class.java).connectionInfo.rssi
        @Suppress("DEPRECATION")
        "wifi" to WifiManager.calculateSignalLevel(rssi, 5)
      }
      caps.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR) -> {
        val level =
          if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P)
            runCatching { context.getSystemService(TelephonyManager::class.java).signalStrength?.level }.getOrNull()
          else null
        "cellular" to (level ?: -1)
      }
      caps.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET) -> "ethernet" to 4
      else -> "other" to -1
    }
  }

  // Media stream, so the kiosk's volume buttons control it and a silenced ringer does not mute punches.
  private val tones by lazy { runCatching { ToneGenerator(AudioManager.STREAM_MUSIC, 90) }.getOrNull() }

  override fun playTone(kind: String) {
    val tone = if (kind == "error") ToneGenerator.TONE_PROP_NACK else ToneGenerator.TONE_PROP_ACK
    tones?.startTone(tone, 300)
  }

  override fun invalidate() {
    tones?.release()
    super.invalidate()
  }

  override fun openSettings(screen: String) {
    val action =
      when (screen) {
        "date" -> Settings.ACTION_DATE_SETTINGS
        "security" -> Settings.ACTION_SECURITY_SETTINGS
        "wifi" -> Settings.ACTION_WIFI_SETTINGS
        else -> Settings.ACTION_SETTINGS
      }
    runCatching { context.startActivity(Intent(action).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)) }
  }
}

class KioskDevicePackage : BaseReactPackage() {
  override fun getModule(name: String, reactContext: ReactApplicationContext): NativeModule? =
    if (name == NativeKioskDeviceSpec.NAME) KioskDeviceModule(reactContext) else null

  override fun getReactModuleInfoProvider() = ReactModuleInfoProvider {
    mapOf(
      NativeKioskDeviceSpec.NAME to
        ReactModuleInfo(NativeKioskDeviceSpec.NAME, NativeKioskDeviceSpec.NAME, false, false, false, true),
    )
  }
}
