# ONNX Runtime JNI bridge — release builds strip this without keep rules
-keep class ai.onnxruntime.** { *; }
-keep class com.microsoft.onnxruntime.** { *; }
