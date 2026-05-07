import { useEffect, useRef, useState } from "react";

export function useAudioRecorder() {
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  useEffect(() => {
    return () => {
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }

      mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, [audioUrl]);

  async function startRecording() {
    setError(null);

    if (!("MediaRecorder" in window)) {
      setError("This browser does not support audio recording.");
      return false;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);

      mediaStreamRef.current = stream;
      mediaRecorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.start();
      setIsRecording(true);
      return true;
    } catch {
      setError("Microphone access was blocked. You can still use the demo flow.");
      return false;
    }
  }

  async function stopRecording() {
    const recorder = mediaRecorderRef.current;

    if (!recorder || recorder.state === "inactive") {
      return audioBlob;
    }

    const blob = await new Promise<Blob>((resolve) => {
      recorder.onstop = () => {
        const nextBlob = new Blob(chunksRef.current, {
          type: recorder.mimeType || "audio/webm",
        });

        resolve(nextBlob);
      };

      recorder.stop();
      mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    });

    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }

    const nextUrl = URL.createObjectURL(blob);
    setAudioBlob(blob);
    setAudioUrl(nextUrl);
    setIsRecording(false);
    return blob;
  }

  function clearRecording() {
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }

    setAudioBlob(null);
    setAudioUrl(null);
  }

  return {
    isRecording,
    audioBlob,
    audioUrl,
    error,
    startRecording,
    stopRecording,
    clearRecording,
  };
}
