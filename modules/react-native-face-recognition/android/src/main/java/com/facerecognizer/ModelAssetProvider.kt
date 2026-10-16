package com.facerecognizer

import android.content.res.AssetManager
import com.margelo.nitro.NitroModules
import java.io.File
import java.io.IOException

private const val MODEL_COPY_BUFFER_BYTES: Int = 64 * 1024

// Intentionally a public object with a public @JvmStatic function: this is
// resolved from C++ via JNI (GetStaticMethodID). Marking it `internal` would
// cause Kotlin to name-mangle the symbol and break the JNI lookup.
object ModelAssetProvider {
  @JvmStatic
  fun materializeBundledModel(assetName: String): String {
    val context = NitroModules.applicationContext
      ?: throw IllegalStateException(
        "FaceRecognizer cannot access the cache directory before NitroModules is initialized.",
      )
    val assets: AssetManager = context.assets
    val cacheDir: File = File(context.cacheDir, "facerecognizer-models").apply { mkdirs() }
    if (!cacheDir.isDirectory) {
      throw IllegalStateException(
        "FaceRecognizer cannot create model cache directory at '${cacheDir.absolutePath}'.",
      )
    }
    val fileName: String = assetName
      .substringAfterLast('/')
      .replace(Regex("[^A-Za-z0-9._-]"), "_")
    val cached: File = File(cacheDir, fileName)
    val assetLength: Long = getAssetLength(assets, assetName)

    if (!cached.exists() || cached.length() != assetLength) {
      streamAssetToCache(assets, assetName, cached, assetLength)
    }
    return cached.absolutePath
  }

  private fun getAssetLength(assets: AssetManager, assetName: String): Long {
    try {
      assets.open(assetName).use { input ->
        return input.available().toLong()
      }
    } catch (error: IOException) {
      throw IllegalStateException(
        "FaceRecognizer cannot read bundled model asset '$assetName'.",
        error,
      )
    }
  }

  private fun streamAssetToCache(
    assets: AssetManager,
    assetName: String,
    cached: File,
    assetLength: Long,
  ) {
    val temporary: File = File(cached.parentFile, "${cached.name}.tmp")
    try {
      assets.open(assetName).use { input ->
        temporary.outputStream().use { output ->
          val buffer: ByteArray = ByteArray(MODEL_COPY_BUFFER_BYTES)
          var bytesRead: Int = input.read(buffer)
          while (bytesRead != -1) {
            output.write(buffer, 0, bytesRead)
            bytesRead = input.read(buffer)
          }
        }
      }
    } catch (error: IOException) {
      throw IllegalStateException(
        "FaceRecognizer cannot copy bundled model asset '$assetName' to '${temporary.absolutePath}'.",
        error,
      )
    }

    if (temporary.length() != assetLength) {
      throw IllegalStateException(
        "FaceRecognizer copied bundled model asset '$assetName' to '${temporary.absolutePath}' with " +
          "${temporary.length()} bytes, expected $assetLength bytes.",
      )
    }
    if (temporary.renameTo(cached)) {
      return
    }
    throw IllegalStateException(
      "FaceRecognizer cannot move streamed model asset '$assetName' from '${temporary.absolutePath}' " +
        "to '${cached.absolutePath}'.",
    )
  }
}
