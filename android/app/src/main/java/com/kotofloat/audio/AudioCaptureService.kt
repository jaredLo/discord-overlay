package com.kotofloat.audio

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Binder
import android.os.Build
import android.os.IBinder
import com.kotofloat.R
import com.kotofloat.asr.WhisperClient
import com.kotofloat.ui.PocActivity
import com.kotofloat.vad.SileroVad
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull

class AudioCaptureService : Service() {

    private val binder = AudioBinder()
    private val _events = MutableEventFlow()
    val events: EventFlow = _events

    private var audioRecord: AudioRecord? = null
    private var vad: SileroVad? = null
    private var whisperClient: WhisperClient? = null
    private var producerJob: Job? = null
    private var consumerJob: Job? = null
    private var whisperWorkerJob: Job? = null
    private var whisperQueue: Channel<Pair<ShortArray, Int>>? = null
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    @Volatile
    private var isRecording = false
    @Volatile
    private var isDraining = false
    private var captureMode = CaptureMode.VAD

    inner class AudioBinder : Binder() {
        fun getService(): AudioCaptureService = this@AudioCaptureService
    }

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // Must call startForeground within 5s of startForegroundService
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(
                NOTIFICATION_ID, createNotification(),
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE
            )
        } else {
            startForeground(NOTIFICATION_ID, createNotification())
        }

        if (!isRecording) {
            val apiKey = intent?.getStringExtra(EXTRA_API_KEY)
            if (apiKey == null) {
                // OS recreated service with null intent after process kill — nothing to resume
                stopSelf()
                return START_NOT_STICKY
            }
            val modeStr = intent.getStringExtra(EXTRA_MODE) ?: "vad"
            captureMode = if (modeStr == "fixed") CaptureMode.FIXED_CHUNK else CaptureMode.VAD
            startCapture(apiKey)
        }

        return START_NOT_STICKY
    }

    private fun startCapture(apiKey: String) {
        if (isDraining) {
            _events.tryEmit(CaptureEvent.Error("Previous session still stopping"))
            return
        }

        whisperClient = WhisperClient(apiKey)

        if (captureMode == CaptureMode.VAD) {
            try {
                vad = SileroVad(this).also { it.reset() }
            } catch (e: Exception) {
                _events.tryEmit(CaptureEvent.Error("Failed to load VAD model: ${e.message}"))
                _events.tryEmit(CaptureEvent.Status("Stopped"))
                stopSelf()
                return
            }
        }

        val bufSize = AudioRecord.getMinBufferSize(
            SileroVad.SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        ).coerceAtLeast(4096) * 2

        @Suppress("MissingPermission") // Permission checked by PocActivity before starting
        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SileroVad.SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufSize
        )

        if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
            _events.tryEmit(CaptureEvent.Error("Failed to initialize AudioRecord"))
            audioRecord?.release()
            audioRecord = null
            vad?.close()
            vad = null
            _events.tryEmit(CaptureEvent.Status("Stopped"))
            stopSelf()
            return
        }

        isRecording = true
        audioRecord?.startRecording()

        // Single-worker Whisper queue — FIFO guarantees transcript order
        whisperQueue = Channel(capacity = 16)
        whisperWorkerJob = scope.launch(Dispatchers.IO) { whisperWorker() }

        when (captureMode) {
            CaptureMode.VAD -> {
                val channel = Channel<AudioWindow>(capacity = 256)
                producerJob = scope.launch(Dispatchers.IO) { runProducer(channel) }
                consumerJob = scope.launch(Dispatchers.Default) { runVadConsumer(channel) }
            }
            CaptureMode.FIXED_CHUNK -> {
                producerJob = scope.launch(Dispatchers.IO) { runFixedChunkLoop() }
            }
        }

        _events.tryEmit(CaptureEvent.Status("Recording (${captureMode.label})"))
    }

    fun stopCapture() {
        if (!isRecording || isDraining) return
        isRecording = false
        isDraining = true

        // Stop mic — AudioRecord.read() returns 0/error, producer exits naturally
        audioRecord?.stop()

        // Drain pipeline: producer → consumer → whisper worker → cleanup → stopSelf
        scope.launch {
            withTimeoutOrNull(3000) {
                producerJob?.join()
                consumerJob?.join()
            }
            producerJob?.cancel()
            consumerJob?.cancel()

            whisperQueue?.close()
            withTimeoutOrNull(35000) { whisperWorkerJob?.join() }
            whisperWorkerJob?.cancel()

            audioRecord?.release()
            audioRecord = null
            vad?.close()
            vad = null
            whisperQueue = null
            producerJob = null
            consumerJob = null
            whisperWorkerJob = null

            stopForeground(STOP_FOREGROUND_REMOVE)
            _events.tryEmit(CaptureEvent.Status("Stopped"))
            isDraining = false
            stopSelf()
        }
    }

    override fun onDestroy() {
        if (isDraining) {
            // Drain coroutine is already handling cleanup — don't cancel its scope
        } else {
            // Framework-initiated teardown or no active session — synchronous cleanup
            isRecording = false
            producerJob?.cancel()
            consumerJob?.cancel()
            whisperWorkerJob?.cancel()
            whisperQueue?.close()
            audioRecord?.stop()
            audioRecord?.release()
            vad?.close()
            scope.cancel()
        }
        super.onDestroy()
    }

    // ── Producer: AudioRecord → 512-sample framed windows → Channel ──

    private suspend fun runProducer(channel: Channel<AudioWindow>) {
        val readBuf = ShortArray(1024)
        val ringBuf = ShortArray(RING_BUF_SIZE)
        var writePos = 0L
        var readPos = 0L

        try {
            while (isRecording && isActive) {
                val read = audioRecord?.read(readBuf, 0, readBuf.size) ?: break
                if (read < 0) break // ERROR_DEAD_OBJECT, ERROR_INVALID_OPERATION, etc.
                if (read == 0) continue

                for (i in 0 until read) {
                    ringBuf[(writePos % RING_BUF_SIZE).toInt()] = readBuf[i]
                    writePos++
                }

                while (writePos - readPos >= SileroVad.WINDOW_SIZE_SAMPLES) {
                    val pcm = ShortArray(SileroVad.WINDOW_SIZE_SAMPLES)
                    val floats = FloatArray(SileroVad.WINDOW_SIZE_SAMPLES)
                    for (i in 0 until SileroVad.WINDOW_SIZE_SAMPLES) {
                        val sample = ringBuf[((readPos + i) % RING_BUF_SIZE).toInt()]
                        pcm[i] = sample
                        floats[i] = sample.toFloat() / 32768f
                    }
                    readPos += SileroVad.WINDOW_SIZE_SAMPLES
                    channel.trySend(AudioWindow(pcm, floats))
                }
            }
        } finally {
            channel.close()
            // If loop exited due to recorder error (not user stop), tear down service
            if (isRecording && !isDraining) {
                _events.tryEmit(CaptureEvent.Error("Microphone error"))
                scope.launch { stopCapture() }
            }
        }
    }

    // ── VAD Consumer: Channel → Silero inference → speech chunks → Whisper ──

    private suspend fun runVadConsumer(channel: Channel<AudioWindow>) {
        val speechChunks = mutableListOf<ShortArray>()
        val shortBuf = mutableListOf<ShortArray>()
        var speechSamples = 0
        var shortBufSamples = 0
        var isSpeaking = false
        var hangover = 0
        var silenceFrames = 0
        var chunkCount = 0

        try {
            for (window in channel) {
                if (!isActive) break

                val prob = vad?.process(window.vadSamples) ?: continue
                val speech = prob >= SileroVad.SPEECH_THRESHOLD

                _events.tryEmit(CaptureEvent.VadUpdate(speech, prob))

                if (speech) {
                    silenceFrames = 0
                    if (!isSpeaking) {
                        isSpeaking = true
                        _events.tryEmit(CaptureEvent.Status("Speech detected"))
                        if (shortBuf.isNotEmpty()) {
                            speechChunks.addAll(shortBuf)
                            speechSamples += shortBufSamples
                            shortBuf.clear()
                            shortBufSamples = 0
                        }
                    }
                    hangover = HANGOVER_FRAMES
                    speechChunks.add(window.pcmSamples)
                    speechSamples += window.pcmSamples.size

                    if (speechSamples >= MAX_CHUNK_SAMPLES) {
                        emitToWhisper(speechChunks, speechSamples, ++chunkCount)
                        speechChunks.clear()
                        speechSamples = 0
                    }
                } else {
                    silenceFrames++

                    if (isSpeaking) {
                        hangover--
                        speechChunks.add(window.pcmSamples) // tail padding
                        speechSamples += window.pcmSamples.size

                        if (hangover <= 0) {
                            isSpeaking = false

                            if (speechSamples < MIN_CHUNK_SAMPLES) {
                                shortBuf.addAll(speechChunks)
                                shortBufSamples += speechSamples
                            } else {
                                emitToWhisper(speechChunks, speechSamples, ++chunkCount)
                            }
                            speechChunks.clear()
                            speechSamples = 0
                            _events.tryEmit(CaptureEvent.Status("Listening..."))
                        }
                    } else if (shortBuf.isNotEmpty() && silenceFrames >= FLUSH_SILENCE_FRAMES) {
                        emitToWhisper(shortBuf, shortBufSamples, ++chunkCount)
                        shortBuf.clear()
                        shortBufSamples = 0
                    }
                }
            }
        } finally {
            // Flush remaining buffered audio on stop
            withContext(NonCancellable) {
                if (speechChunks.isNotEmpty()) {
                    emitToWhisper(speechChunks, speechSamples, ++chunkCount)
                }
                if (shortBuf.isNotEmpty()) {
                    emitToWhisper(shortBuf, shortBufSamples, ++chunkCount)
                }
            }
        }
    }

    /** Enqueue audio for Whisper. Single worker processes FIFO → ordered transcripts. */
    private suspend fun emitToWhisper(chunks: List<ShortArray>, totalSamples: Int, chunkNum: Int) {
        val pcm = flattenChunks(chunks, totalSamples)
        try {
            whisperQueue?.send(pcm to chunkNum)
        } catch (_: Exception) {
            // Queue closed during shutdown — acceptable to drop
        }
    }

    private suspend fun whisperWorker() {
        val queue = whisperQueue ?: return
        for ((pcm, chunkNum) in queue) {
            processChunk(pcm, chunkNum)
        }
    }

    // ── Fixed-chunk mode: 3-second blocks, no VAD ──

    private suspend fun runFixedChunkLoop() {
        val samplesFor3s = SileroVad.SAMPLE_RATE * 3
        val readBuf = ShortArray(1024)
        val chunkBuf = ShortArray(samplesFor3s)
        var offset = 0
        var chunkCount = 0

        try {
            while (isRecording && isActive) {
                val read = audioRecord?.read(readBuf, 0, readBuf.size) ?: break
                if (read < 0) break // ERROR_DEAD_OBJECT, ERROR_INVALID_OPERATION, etc.
                if (read == 0) continue

                val toCopy = minOf(read, samplesFor3s - offset)
                readBuf.copyInto(chunkBuf, offset, 0, toCopy)
                offset += toCopy

                if (offset >= samplesFor3s) {
                    emitToWhisper(listOf(chunkBuf.copyOf()), samplesFor3s, ++chunkCount)
                    offset = 0

                    if (toCopy < read) {
                        readBuf.copyInto(chunkBuf, 0, toCopy, read)
                        offset = read - toCopy
                    }
                }
            }
        } finally {
            // Flush partial buffer on stop
            withContext(NonCancellable) {
                if (offset > 0) {
                    emitToWhisper(listOf(chunkBuf.copyOfRange(0, offset)), offset, ++chunkCount)
                }
            }
            // If loop exited due to recorder error, tear down service
            if (isRecording && !isDraining) {
                _events.tryEmit(CaptureEvent.Error("Microphone error"))
                scope.launch { stopCapture() }
            }
        }
    }

    // ── Whisper transcription ──

    private suspend fun processChunk(pcm: ShortArray, chunkNum: Int) {
        val durationSec = pcm.size.toFloat() / SileroVad.SAMPLE_RATE
        _events.tryEmit(
            CaptureEvent.Status("Transcribing #$chunkNum (${String.format("%.1f", durationSec)}s)...")
        )

        val result = withContext(Dispatchers.IO) { whisperClient?.transcribe(pcm) }

        if (result?.text != null) {
            _events.tryEmit(CaptureEvent.Transcript(result.text, durationSec, chunkNum))
        } else {
            val reason = result?.error ?: "No response"
            _events.tryEmit(CaptureEvent.Error("Chunk #$chunkNum failed: $reason"))
        }
    }

    // ── Helpers ──

    private fun flattenChunks(chunks: List<ShortArray>, totalSize: Int): ShortArray {
        val result = ShortArray(totalSize)
        var offset = 0
        for (chunk in chunks) {
            chunk.copyInto(result, offset)
            offset += chunk.size
        }
        return result
    }

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID, "Audio Capture", NotificationManager.IMPORTANCE_LOW
        ).apply { description = "KotoFloat audio capture" }
        getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
    }

    private fun createNotification(): Notification {
        val intent = Intent(this, PocActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP
        }
        val pendingIntent = PendingIntent.getActivity(
            this, 0, intent, PendingIntent.FLAG_IMMUTABLE
        )
        return Notification.Builder(this, CHANNEL_ID)
            .setContentTitle(getString(R.string.app_name))
            .setContentText(getString(R.string.notification_recording))
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .build()
    }

    companion object {
        const val EXTRA_API_KEY = "api_key"
        const val EXTRA_MODE = "capture_mode"

        private const val CHANNEL_ID = "kotofloat_capture"
        private const val NOTIFICATION_ID = 1

        private const val RING_BUF_SIZE = 16384       // ~1s buffer at 16kHz
        private const val HANGOVER_FRAMES = 10         // 320ms (10 * 32ms per frame)
        private const val MIN_CHUNK_SAMPLES = 6400     // 0.4s at 16kHz
        private const val MAX_CHUNK_SAMPLES = 480000   // 30s at 16kHz
        private const val FLUSH_SILENCE_FRAMES = 94    // ~3s (94 * 32ms)
    }
}

// ── Data types ──

data class AudioWindow(
    val pcmSamples: ShortArray,
    val vadSamples: FloatArray,
)

enum class CaptureMode(val label: String) {
    VAD("VAD"), FIXED_CHUNK("Fixed 3s")
}

sealed class CaptureEvent {
    data class Transcript(val text: String, val durationSec: Float, val chunkNum: Int) : CaptureEvent()
    data class Status(val message: String) : CaptureEvent()
    data class Error(val message: String) : CaptureEvent()
    data class VadUpdate(val isSpeech: Boolean, val probability: Float) : CaptureEvent()
}

// Type aliases for SharedFlow usage
private typealias MutableEventFlow = kotlinx.coroutines.flow.MutableSharedFlow<CaptureEvent>
private typealias EventFlow = kotlinx.coroutines.flow.SharedFlow<CaptureEvent>

private fun MutableEventFlow(): MutableEventFlow =
    kotlinx.coroutines.flow.MutableSharedFlow(replay = 16, extraBufferCapacity = 64)
