package com.kotofloat.vad

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * Silero VAD v4 wrapper using ONNX Runtime.
 * Processes 512-sample (32ms) float32 windows at 16kHz.
 * Carries hidden state (h/c) between calls for temporal context.
 */
class SileroVad(context: Context) {

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val hState = FloatArray(STATE_SIZE)
    private val cState = FloatArray(STATE_SIZE)

    init {
        val modelBytes = context.assets.open(MODEL_FILE).readBytes()
        session = env.createSession(modelBytes)
    }

    /**
     * Run VAD on a 512-sample float32 window (normalized to [-1.0, 1.0]).
     * Returns speech probability [0.0, 1.0].
     */
    fun process(samples: FloatArray): Float {
        require(samples.size == WINDOW_SIZE_SAMPLES) {
            "Expected $WINDOW_SIZE_SAMPLES samples, got ${samples.size}"
        }

        val tensors = mutableListOf<OnnxTensor>()
        try {
            val input = OnnxTensor.createTensor(
                env, FloatBuffer.wrap(samples), longArrayOf(1, WINDOW_SIZE_SAMPLES.toLong())
            ).also { tensors.add(it) }

            val sr = OnnxTensor.createTensor(
                env, LongBuffer.wrap(longArrayOf(SAMPLE_RATE.toLong())), longArrayOf(1)
            ).also { tensors.add(it) }

            val h = OnnxTensor.createTensor(
                env, FloatBuffer.wrap(hState), longArrayOf(2, 1, 64)
            ).also { tensors.add(it) }

            val c = OnnxTensor.createTensor(
                env, FloatBuffer.wrap(cState), longArrayOf(2, 1, 64)
            ).also { tensors.add(it) }

            val inputs = mapOf("input" to input, "sr" to sr, "h" to h, "c" to c)

            session.run(inputs).use { result ->
                @Suppress("UNCHECKED_CAST")
                val prob = (result.get(0).value as Array<FloatArray>)[0][0]

                // Carry hidden state forward
                @Suppress("UNCHECKED_CAST")
                flatten3d(result.get(1).value as Array<Array<FloatArray>>, hState)
                @Suppress("UNCHECKED_CAST")
                flatten3d(result.get(2).value as Array<Array<FloatArray>>, cState)

                return prob
            }
        } finally {
            tensors.forEach { it.close() }
        }
    }

    /** Reset hidden state — call on session start or reconnect. */
    fun reset() {
        hState.fill(0f)
        cState.fill(0f)
    }

    fun close() {
        session.close()
    }

    private fun flatten3d(src: Array<Array<FloatArray>>, dst: FloatArray) {
        var i = 0
        for (a in src) for (b in a) for (v in b) dst[i++] = v
    }

    companion object {
        const val WINDOW_SIZE_SAMPLES = 512  // 32ms at 16kHz
        const val SAMPLE_RATE = 16000
        const val SPEECH_THRESHOLD = 0.5f
        const val MODEL_FILE = "silero_vad.onnx"
        private const val STATE_SIZE = 2 * 1 * 64  // [2, 1, 64] flattened
    }
}
