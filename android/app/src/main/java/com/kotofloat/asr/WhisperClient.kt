package com.kotofloat.asr

import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.TimeUnit

/**
 * Direct OpenAI Whisper API client for Phase 0 PoC.
 * Takes raw PCM int16 audio, wraps in WAV, sends to transcription endpoint.
 */
class WhisperClient(private val apiKey: String) {

    private val client = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    /** Transcribe PCM int16 audio. Returns Japanese text or null on failure. */
    fun transcribe(pcm: ShortArray): TranscriptResult? {
        if (pcm.isEmpty()) return null

        val wavBytes = pcmToWav(pcm)

        val body = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart(
                "file", "audio.wav",
                wavBytes.toRequestBody("audio/wav".toMediaType())
            )
            .addFormDataPart("model", MODEL)
            .addFormDataPart("language", "ja")
            .addFormDataPart("response_format", "json")
            .build()

        val request = Request.Builder()
            .url(WHISPER_URL)
            .header("Authorization", "Bearer $apiKey")
            .post(body)
            .build()

        return try {
            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    val errorBody = response.body?.string()
                    return TranscriptResult(
                        text = null,
                        error = "HTTP ${response.code}: $errorBody"
                    )
                }
                val json = JSONObject(response.body?.string() ?: return null)
                val text = json.optString("text").takeIf { it.isNotBlank() }
                TranscriptResult(text = text, error = null)
            }
        } catch (e: Exception) {
            TranscriptResult(text = null, error = e.message ?: "Unknown error")
        }
    }

    data class TranscriptResult(val text: String?, val error: String?)

    companion object {
        private const val WHISPER_URL = "https://api.openai.com/v1/audio/transcriptions"
        private const val MODEL = "gpt-4o-mini-transcribe"

        /** Wrap raw PCM int16 samples in a WAV container with RIFF headers. */
        fun pcmToWav(pcm: ShortArray, sampleRate: Int = 16000): ByteArray {
            val dataSize = pcm.size * 2 // 2 bytes per int16 sample
            val buffer = ByteBuffer.allocate(44 + dataSize).order(ByteOrder.LITTLE_ENDIAN)

            // RIFF header
            buffer.put("RIFF".toByteArray(Charsets.US_ASCII))
            buffer.putInt(36 + dataSize)
            buffer.put("WAVE".toByteArray(Charsets.US_ASCII))

            // fmt chunk
            buffer.put("fmt ".toByteArray(Charsets.US_ASCII))
            buffer.putInt(16)              // chunk size
            buffer.putShort(1)             // PCM format
            buffer.putShort(1)             // mono
            buffer.putInt(sampleRate)      // sample rate
            buffer.putInt(sampleRate * 2)  // byte rate (sampleRate * channels * bytesPerSample)
            buffer.putShort(2)             // block align (channels * bytesPerSample)
            buffer.putShort(16)            // bits per sample

            // data chunk
            buffer.put("data".toByteArray(Charsets.US_ASCII))
            buffer.putInt(dataSize)
            for (s in pcm) buffer.putShort(s)

            return buffer.array()
        }
    }
}
