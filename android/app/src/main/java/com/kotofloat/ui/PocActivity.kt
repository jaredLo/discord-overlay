package com.kotofloat.ui

import android.Manifest
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.kotofloat.audio.AudioCaptureService
import com.kotofloat.audio.CaptureEvent
import com.kotofloat.audio.CaptureMode
import com.kotofloat.databinding.ActivityPocBinding
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch

class PocActivity : AppCompatActivity() {

    private lateinit var binding: ActivityPocBinding
    private var captureService: AudioCaptureService? = null
    private var isBound = false
    private var isRecording = false
    private var collectJob: Job? = null

    private val prefs by lazy { getSharedPreferences("kotofloat_poc", Context.MODE_PRIVATE) }

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { results ->
        if (results[Manifest.permission.RECORD_AUDIO] == true) {
            startCapture()
        } else {
            Toast.makeText(this, "Microphone permission required", Toast.LENGTH_SHORT).show()
        }
    }

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            captureService = (service as AudioCaptureService.AudioBinder).getService()
            isBound = true
            collectEvents()
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            captureService = null
            isBound = false
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityPocBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.apiKeyInput.setText(prefs.getString("api_key", ""))

        binding.toggleButton.setOnClickListener {
            if (isRecording) stopCapture() else requestPermissionsAndStart()
        }

        binding.clearButton.setOnClickListener {
            binding.transcriptText.text = ""
        }
    }

    private fun requestPermissionsAndStart() {
        val needed = mutableListOf<String>()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            needed.add(Manifest.permission.RECORD_AUDIO)
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
            != PackageManager.PERMISSION_GRANTED
        ) {
            needed.add(Manifest.permission.POST_NOTIFICATIONS)
        }

        if (needed.isEmpty()) {
            startCapture()
        } else {
            permissionLauncher.launch(needed.toTypedArray())
        }
    }

    private fun startCapture() {
        val apiKey = binding.apiKeyInput.text?.toString()?.trim()
        if (apiKey.isNullOrBlank()) {
            Toast.makeText(this, "Enter your OpenAI API key", Toast.LENGTH_SHORT).show()
            return
        }
        prefs.edit().putString("api_key", apiKey).apply()

        val mode = if (binding.modeVad.isChecked) "vad" else "fixed"

        val intent = Intent(this, AudioCaptureService::class.java).apply {
            putExtra(AudioCaptureService.EXTRA_API_KEY, apiKey)
            putExtra(AudioCaptureService.EXTRA_MODE, mode)
        }
        ContextCompat.startForegroundService(this, intent)
        bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE)

        isRecording = true
        binding.toggleButton.text = "Stop Recording"
        binding.apiKeyInput.isEnabled = false
        binding.modeGroup.isEnabled = false
        binding.modeVad.isEnabled = false
        binding.modeFixed.isEnabled = false
        binding.statusText.text = "Starting..."
    }

    private fun collectEvents() {
        collectJob = lifecycleScope.launch {
            captureService?.events?.collect { event ->
                when (event) {
                    is CaptureEvent.Transcript -> {
                        appendLine("[#${event.chunkNum} ${String.format("%.1f", event.durationSec)}s] ${event.text}")
                    }
                    is CaptureEvent.Status -> {
                        binding.statusText.text = event.message
                        if (event.message == "Stopped") onSessionEnded()
                    }
                    is CaptureEvent.Error -> {
                        appendLine("ERR: ${event.message}")
                        binding.statusText.text = "Error"
                    }
                    is CaptureEvent.VadUpdate -> {
                        val dot = if (event.isSpeech) "\uD83D\uDD34" else "\u26AA"
                        binding.vadIndicator.text = "$dot ${String.format("%.2f", event.probability)}"
                    }
                }
            }
        }
    }

    private fun appendLine(line: String) {
        binding.transcriptText.append("$line\n\n")
        binding.transcriptScroll.post {
            binding.transcriptScroll.fullScroll(android.view.View.FOCUS_DOWN)
        }
    }

    private fun stopCapture() {
        captureService?.stopCapture()
        // Stay subscribed — service drains buffered audio and emits final transcripts.
        // onSessionEnded() is called when "Stopped" status event arrives.
    }

    /** Called on "Stopped" event — handles both user-initiated stop and error-triggered stop. */
    private fun onSessionEnded() {
        isRecording = false
        binding.toggleButton.text = "Start Recording"
        binding.apiKeyInput.isEnabled = true
        binding.modeGroup.isEnabled = true
        binding.modeVad.isEnabled = true
        binding.modeFixed.isEnabled = true
        detachFromService()
    }

    private fun detachFromService() {
        collectJob?.cancel()
        if (isBound) {
            unbindService(serviceConnection)
            isBound = false
        }
        captureService = null
        binding.vadIndicator.text = ""
    }

    override fun onDestroy() {
        collectJob?.cancel()
        if (isBound) {
            unbindService(serviceConnection)
            isBound = false
        }
        super.onDestroy()
    }
}
