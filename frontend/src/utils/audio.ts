export async function playAudioSource(
  url?: string | null,
  fallbackText?: string,
  language = "en-US",
) {
  if (url) {
    const audio = new Audio(url);

    try {
      await audio.play();
      return;
    } catch {
      if (!fallbackText) {
        throw new Error("Audio playback failed.");
      }
    }
  }

  if (fallbackText && "speechSynthesis" in window) {
    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(fallbackText);
    utterance.lang = language;
    utterance.rate = 0.85;
    utterance.pitch = 1;
    window.speechSynthesis.speak(utterance);
  }
}
